"""Release implementation of MaDiS."""

import os
import pickle
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3ForCausalLM

from mGPT.archs.connectors import MLPConnector
from mGPT.config import instantiate_from_config
from mGPT.losses.mgpt import GPTLosses, clip_loss, create_mask
from mGPT.models.base import BaseModel
from mGPT.utils.load_checkpoint import load_pretrained_vae


class MaDiS(BaseModel):
    """Masked-diffusion sign generator and SiCLIP evaluator."""

    def __init__(
        self,
        cfg,
        datamodule,
        lm,
        motion_vae,
        hand_vae_cfg,
        rhand_vae_cfg,
        stage,
        metrics_dict,
        max_gen_len=100,
        **_,
    ):
        self.save_hyperparameters(
            ignore=(
                "datamodule", "lm", "motion_vae", "hand_vae_cfg",
                "rhand_vae_cfg",
            ),
            logger=False,
        )
        self.datamodule = datamodule
        super().__init__()

        self.vae = instantiate_from_config(motion_vae)
        self.hand_vae = instantiate_from_config(hand_vae_cfg)
        self.rhand_vae = instantiate_from_config(rhand_vae_cfg)
        self.max_gen_len = max_gen_len
        self.dim_per_joint = 3

        if cfg.TRAIN.PRETRAINED_VAE:
            load_pretrained_vae(cfg, self)
        for tokenizer in (self.vae, self.hand_vae, self.rhand_vae):
            tokenizer.eval()
            tokenizer.requires_grad_(False)

        if stage == "lm_clip":
            # SiCLIP learns its sign encoders jointly with the projection heads.
            for tokenizer in (self.vae, self.hand_vae, self.rhand_vae):
                tokenizer.encoder.train()
                tokenizer.encoder.requires_grad_(True)

        if stage in {"lm_pretrain", "lm_instruct"}:
            lm = deepcopy(lm)
            lm["params"]["motion_codebook_size"] = self.vae.code_num
            lm["params"]["hand_codebook_size"] = self.hand_vae.code_num
            lm["params"]["rhand_codebook_size"] = self.rhand_vae.code_num
            lm["params"]["body_codebook"] = (
                self.vae.quantizer.codebook.detach().clone())
            lm["params"]["hand_codebook"] = (
                self.hand_vae.quantizer.codebook.detach().clone())
            lm["params"]["rhand_codebook"] = (
                self.rhand_vae.quantizer.codebook.detach().clone())
            lm["params"]["stage"] = stage
            self.lm = instantiate_from_config(lm)

            hidden_size = self.lm.language_model.main_lm.config.hidden_size
            intermediate_size = self.lm.language_model.main_lm.config.intermediate_size
            self.mlp_body = MLPConnector(
                hidden_size, intermediate_size, self.vae.code_dim)
            self.mlp_lhand = MLPConnector(
                hidden_size, intermediate_size, self.hand_vae.code_dim)
            self.mlp_rhand = MLPConnector(
                hidden_size, intermediate_size, self.rhand_vae.code_dim)
        elif stage == "lm_clip":
            model_path = lm["params"]["model_path"]
            self.lm = Qwen3ForCausalLM.from_pretrained(
                model_path, num_hidden_layers=14, output_hidden_states=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
            with open(os.path.join(model_path, "map_ids.pkl"), "rb") as handle:
                self.map_ids = pickle.load(handle)
            self.pad_idx = self.map_ids[self.tokenizer.pad_token_id]
            self.text_proj = MLPConnector(
                self.lm.config.hidden_size,
                self.lm.config.intermediate_size,
                self.vae.code_dim,
            )
            self.pose_proj = MLPConnector(
                self.vae.code_dim * 3,
                self.lm.config.intermediate_size,
                self.vae.code_dim,
            )
            self.logit_scale = nn.Parameter(torch.randn(1))
            self.logit_bias = torch.zeros(1)

        self._losses = nn.ModuleDict({
            f"losses_{split}": GPTLosses(cfg, stage, datamodule.njoints)
            for split in ("train", "val", "test")
        })
        self.feats2joints = datamodule.feats2joints

    def train_lm_forward(self, batch, split):
        tokens = batch["m_tokens"]
        token_lengths = batch["m_token_len"]
        texts = batch["text"]

        outputs = self.lm(
            texts,
            tokens,
            token_lengths,
            batch["tasks"],
            src=batch["src"],
            name=batch["name"],
        )
        hidden_states = outputs.pop("hidden_states")
        text_length = outputs.pop("text_len")
        mask_indices = outputs.pop("mask_indices")

        if split == "train":
            if self.hparams.stage == "lm_pretrain":
                mask_indices = mask_indices[:, text_length:]
                hidden_states = hidden_states[:, text_length:]

            valid = torch.arange(
                mask_indices.shape[1], device=mask_indices.device
            ).unsqueeze(0) < torch.as_tensor(
                token_lengths, device=mask_indices.device).unsqueeze(1)
            masked_motion = mask_indices & valid

            latent_body = self.mlp_body(hidden_states)
            latent_lhand = self.mlp_lhand(hidden_states)
            latent_rhand = self.mlp_rhand(hidden_states)
            pose_body = self.vae.decoder(
                latent_body.transpose(1, 2)).transpose(1, 2)
            pose_lhand = self.hand_vae.decoder(
                latent_lhand.transpose(1, 2)).transpose(1, 2)
            pose_rhand = self.rhand_vae.decoder(
                latent_rhand.transpose(1, 2)).transpose(1, 2)
            reconstructed = torch.cat([
                pose_body[..., :10 * self.dim_per_joint],
                pose_lhand,
                pose_rhand,
                pose_body[..., 10 * self.dim_per_joint:],
            ], dim=-1)

            outputs["recons_motions"] = reconstructed
            outputs["gt_motions"] = batch["motion"]

            masked_tokens = tokens[masked_motion].long()
            outputs["gt_latent"] = torch.cat([
                self.vae.quantizer.codebook.detach()[masked_tokens[..., 0]],
                self.hand_vae.quantizer.codebook.detach()[masked_tokens[..., 1]],
                self.rhand_vae.quantizer.codebook.detach()[masked_tokens[..., 2]],
            ], dim=-1)
            outputs["recons_latent"] = torch.cat([
                latent_body, latent_lhand, latent_rhand
            ], dim=-1)[masked_motion]

        return {"outputs": outputs}

    @torch.no_grad()
    def generate_forward(self, batch):
        reference = batch["motion"]
        batch_size, _, feature_size = reference.shape
        generated = self.lm.generate_conditional(
            batch["text"],
            lengths=batch["m_token_len"],
            tasks=None,
            src=batch["src"],
            max_length=self.max_gen_len,
        )
        body_tokens = generated["outputs_tokens"]
        lhand_tokens = generated["outputs_tokens_hand"]
        rhand_tokens = generated["outputs_tokens_rhand"]
        token_count = max(
            max(map(len, body_tokens)),
            max(map(len, lhand_tokens)),
            max(map(len, rhand_tokens)),
        )
        max_frames = token_count * 4
        result = torch.zeros(
            batch_size, max_frames, feature_size, device=reference.device,
            dtype=reference.dtype)
        result_lengths = list(batch["length"])

        for index in range(batch_size):
            body = body_tokens[index].clamp(0, self.vae.code_num - 1)
            lhand = lhand_tokens[index].clamp(0, self.hand_vae.code_num - 1)
            rhand = rhand_tokens[index].clamp(0, self.rhand_vae.code_num - 1)

            body_motion = self.vae.decode(body)
            lhand_motion = self.hand_vae.decode(lhand)
            rhand_motion = self.rhand_vae.decode(rhand)
            result_lengths[index] = max(
                body_motion.shape[1],
                lhand_motion.shape[1],
                rhand_motion.shape[1],
            )

            def pad(motion):
                return F.pad(
                    motion, (0, 0, 0, max_frames - motion.shape[1]),
                    mode="replicate")

            body_motion = pad(body_motion)
            result[index:index + 1, :, :10 * self.dim_per_joint] = (
                body_motion[..., :10 * self.dim_per_joint])
            result[index:index + 1, :, -10 - self.dim_per_joint:] = (
                body_motion[..., 10 * self.dim_per_joint:])
            result[
                index:index + 1,
                :,
                10 * self.dim_per_joint:25 * self.dim_per_joint,
            ] = pad(lhand_motion)
            result[
                index:index + 1,
                :,
                25 * self.dim_per_joint:40 * self.dim_per_joint,
            ] = pad(rhand_motion)

        vertices_ref, joints_ref = self.feats2joints(reference)
        vertices_rst, joints_rst = self.feats2joints(result)
        return {
            "m_ref": reference,
            "m_rst": result,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "vertices_ref": vertices_ref,
            "vertices_rst": vertices_rst,
            "lengths_rst": result_lengths,
            "body_tokens": body_tokens,
            "lhand_tokens": lhand_tokens,
            "rhand_tokens": rhand_tokens,
            "tokens_ref": batch["m_tokens"],
            "time_cost": generated["time_cost"],
        }

    def get_clip_logits(self, batch):
        motion = batch["motion"]
        device = motion.device
        inputs = self.tokenizer(
            batch["text"],
            padding="longest",
            max_length=256,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_length=True,
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        for row in range(input_ids.shape[0]):
            for column in range(input_ids.shape[1]):
                input_ids[row, column] = self.map_ids.get(
                    input_ids[row, column].item(), self.pad_idx)

        text_features = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).hidden_states[-1]
        frame_lengths = [length // 4 for length in batch["length"]]
        pose_mask = create_mask(frame_lengths, device)

        lhand = motion[..., 10 * self.dim_per_joint:25 * self.dim_per_joint]
        rhand = motion[..., 25 * self.dim_per_joint:40 * self.dim_per_joint]
        body = torch.cat([
            motion[..., :10 * self.dim_per_joint],
            motion[..., 40 * self.dim_per_joint:],
        ], dim=-1)
        body = self.vae.encoder(body.transpose(1, 2)).transpose(1, 2)
        lhand = self.hand_vae.encoder(lhand.transpose(1, 2)).transpose(1, 2)
        rhand = self.rhand_vae.encoder(rhand.transpose(1, 2)).transpose(1, 2)
        pose_features = torch.cat([body, lhand, rhand], dim=-1)

        text_features = F.normalize(self.text_proj(text_features), dim=-1)
        pose_features = F.normalize(self.pose_proj(pose_features), dim=-1)
        token_similarity = torch.matmul(
            text_features.unsqueeze(1),
            pose_features.unsqueeze(0).transpose(-1, -2),
        )
        token_similarity = token_similarity * self.logit_scale.exp()

        text_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
        motion_mask = pose_mask.transpose(1, 2).unsqueeze(0)
        combined_mask = text_mask * motion_mask
        masked_similarity = token_similarity.masked_fill(
            combined_mask == 0, -1e6)

        pose_weights = F.softmax(masked_similarity, dim=-1)
        logits_per_text = (pose_weights * token_similarity).sum(dim=-1)
        logits_per_text = (
            logits_per_text * attention_mask.unsqueeze(1)
        ).sum(dim=-1) / attention_mask.sum(dim=-1, keepdim=True)

        text_weights = F.softmax(masked_similarity, dim=-2)
        logits_per_pose = (text_weights * token_similarity).sum(dim=-2)
        valid_motion = motion_mask.squeeze(0)
        logits_per_pose = (
            logits_per_pose * valid_motion
        ).sum(dim=-1) / valid_motion.sum(dim=-1).clamp_min(1)
        return logits_per_text, logits_per_pose

    def train_clip_forward(self, batch):
        logits_per_text, logits_per_pose = self.get_clip_logits(batch)
        return {"clip_loss": (
            clip_loss(logits_per_text) + clip_loss(logits_per_pose)) / 2}

    @torch.no_grad()
    def val_clip_forward(self, batch):
        _, logits_per_pose = self.get_clip_logits(batch)
        return logits_per_pose / self.logit_scale.exp()

    def allsplit_step(self, split, batch, batch_idx):
        del batch_idx
        stage = self.hparams.stage
        loss = None
        if stage in {"lm_pretrain", "lm_instruct"} and split in {"train", "val"}:
            result = self.train_lm_forward(batch, split)
            loss = self._losses[f"losses_{split}"].update(result)
        elif stage == "lm_clip" and split in {"train", "val"}:
            result = self.train_clip_forward(batch)
            loss = self._losses[f"losses_{split}"].update(result)

        if split in {"val", "test"}:
            if stage == "lm_instruct":
                result = self.generate_forward(batch)
                self.metrics.GenerationMetrics.update(
                    feats_rst=result["m_rst"],
                    feats_ref=result["m_ref"],
                    joints_rst=result["joints_rst"],
                    joints_ref=result["joints_ref"],
                    vertices_rst=result["vertices_rst"],
                    vertices_ref=result["vertices_ref"],
                    body_tokens=result["body_tokens"],
                    lhand_tokens=result["lhand_tokens"],
                    rhand_tokens=result["rhand_tokens"],
                    tokens_ref=result["tokens_ref"].long(),
                    lengths=batch["length"],
                    lengths_rst=result["lengths_rst"],
                    time_cost=result["time_cost"],
                    split=split,
                    src=batch["src"],
                    name=batch["name"],
                    is_discrete=True,
                )
            elif stage == "lm_clip":
                similarity = self.val_clip_forward(batch)
                self.metrics.CLIPMetrics.update(
                    sim=similarity, src=batch["src"], name=batch["name"])

        if split == "test":
            if stage == "lm_pretrain":
                raise RuntimeError(
                    "The pretraining checkpoint is validated with token losses; "
                    "use an lm_instruct checkpoint for generation evaluation"
                )
            if stage == "lm_clip":
                return {
                    "name": batch["name"],
                    "sim": similarity,
                    "text": batch["text"],
                }
            return {
                "name": batch["name"],
                "feats_ref": result["m_ref"],
                "feats_rst": result["m_rst"],
                "lengths": batch["length"],
                "lengths_rst": result["lengths_rst"],
                "text": batch["text"],
            }
        return loss
