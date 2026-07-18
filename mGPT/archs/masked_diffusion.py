"""Masked diffusion language model used by MaDiS.

This module implements the method described in the paper: bidirectional
masked-token prediction, tri-level pretraining, unmasking with temporal
checkpoints (UTC), and mixture-of-parts (MoP) embeddings.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import Qwen3ForCausalLM

from mGPT.archs.connectors import MLPConnector


UTC_TRAINING_PROBABILITY = 1.0
UTC_PIVOTS = (0.75, 0.5)
DIFFUSION_STEPS = 25


def _bidirectional_attention_mask(mask_2d: torch.Tensor, dtype: torch.dtype):
    """Expand a padding mask without imposing a causal constraint."""
    valid = mask_2d.unsqueeze(1)
    mask_4d = valid.unsqueeze(-1) * valid.unsqueeze(-2)
    min_value = torch.finfo(dtype).min
    return torch.where(mask_4d.to(torch.bool), 0.0, min_value).to(dtype)


class MaskedDiffusionLanguageModel(nn.Module):
    """Qwen3-based masked diffusion model for part-wise sign tokens."""

    def __init__(
        self,
        model_path: str,
        len_token: int,
        ids_remove_motion,
        ids_remove_hand,
        ids_remove_rhand,
        ids_motion,
        ids_hand,
        ids_rhand,
        pad_idx: int,
        eos_idx: int,
        mask_idx: int,
        body_codebook: torch.Tensor,
        hand_codebook: torch.Tensor,
        rhand_codebook: torch.Tensor,
        stage: str,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.mask_idx = mask_idx
        self.stage = stage
        self.pretraining = stage == "lm_pretrain"

        language_model = Qwen3ForCausalLM.from_pretrained(model_path)
        language_model.resize_token_embeddings(len_token)
        hidden_size = language_model.config.hidden_size

        self.final_logits_bias = torch.tensor([0.0])
        self.main_lm = language_model.get_decoder()

        self.id_dict = {
            "body": ids_motion,
            "lhand": ids_hand,
            "rhand": ids_rhand,
        }
        self.mask_body = torch.zeros(len_token)
        self.mask_body[ids_remove_motion] = float("-inf")
        self.mask_lhand = torch.zeros(len_token)
        self.mask_lhand[ids_remove_hand] = float("-inf")
        self.mask_rhand = torch.zeros(len_token)
        self.mask_rhand[ids_remove_rhand] = float("-inf")
        self.mask_pretrain = torch.zeros(len_token)
        self.mask_pretrain[self.mask_idx] = float("-inf")

        if body_codebook is None or hand_codebook is None or rhand_codebook is None:
            raise ValueError("MaDiS requires body, left-hand, and right-hand codebooks")
        self.register_buffer("body_codebook", body_codebook)
        self.register_buffer("lhand_codebook", hand_codebook)
        self.register_buffer("rhand_codebook", rhand_codebook)

        # Names and shapes are kept compatible with the released checkpoints.
        self.body_emb_mapper = nn.Linear(
            body_codebook.shape[-1], hidden_size, bias=False)
        self.lhand_emb_mapper = nn.Linear(
            hand_codebook.shape[-1], hidden_size, bias=False)
        self.rhand_emb_mapper = nn.Linear(
            rhand_codebook.shape[-1], hidden_size, bias=False)
        self.gate_net = MLPConnector(
            body_codebook.shape[-1] * 3, hidden_size, 3, dropout=0.1)

    def _utc_mask(self, tokens: torch.Tensor, mask_probability: torch.Tensor):
        """Sample only states compatible with UTC's two temporal pivots."""
        batch_size, sequence_length = tokens.shape[:2]
        device = tokens.device
        positions = torch.arange(sequence_length, device=device)
        quarter_grid = (positions % 4 == 0).expand(batch_size, -1)
        half_grid = (positions % 2 == 0).expand(batch_size, -1)
        remaining_grid = ~half_grid

        high_noise = mask_probability >= UTC_PIVOTS[0]
        medium_noise = (
            (mask_probability >= UTC_PIVOTS[1])
            & (mask_probability < UTC_PIVOTS[0])
        )
        low_noise = mask_probability < UTC_PIVOTS[1]
        kept = torch.zeros(
            (batch_size, sequence_length), dtype=torch.bool, device=device)

        if high_noise.any():
            probability = (
                (mask_probability[high_noise] - UTC_PIVOTS[0])
                / (1.0 - UTC_PIVOTS[0])
            ).unsqueeze(1)
            randomly_masked = torch.rand(
                (high_noise.sum(), sequence_length), device=device
            ) < probability
            kept[high_noise] = quarter_grid[high_noise] & ~randomly_masked

        if medium_noise.any():
            probability = (
                (mask_probability[medium_noise] - UTC_PIVOTS[1])
                / (UTC_PIVOTS[0] - UTC_PIVOTS[1])
            ).unsqueeze(1)
            randomly_masked = torch.rand(
                (medium_noise.sum(), sequence_length), device=device
            ) < probability
            kept[medium_noise] = quarter_grid[medium_noise] | (
                half_grid[medium_noise] & ~randomly_masked)

        if low_noise.any():
            probability = (
                mask_probability[low_noise] / UTC_PIVOTS[1]
            ).unsqueeze(1)
            randomly_masked = torch.rand(
                (low_noise.sum(), sequence_length), device=device
            ) < probability
            kept[low_noise] = half_grid[low_noise] | (
                remaining_grid[low_noise] & ~randomly_masked)

        noisy_tokens = tokens.clone()
        noisy_tokens[~kept] = self.mask_idx
        return noisy_tokens

    def _add_noise(self, tokens: torch.Tensor, eps: float = 1e-3):
        batch_size, sequence_length = tokens.shape[:2]
        time = torch.rand(batch_size, device=tokens.device)
        mask_probability = (1.0 - eps) * time + eps

        # UTC is a fixed part of supervised fine-tuning.  The 0.2 mixture is
        # the paper setting, not a user-facing feature flag.
        if (
            not self.pretraining
            and torch.rand((), device=tokens.device)
            < UTC_TRAINING_PROBABILITY
        ):
            noisy_tokens = self._utc_mask(tokens, mask_probability)
        else:
            selected = torch.rand(
                (batch_size, sequence_length), device=tokens.device
            ) < mask_probability.unsqueeze(1)
            for _ in range(tokens.ndim - 2):
                selected = selected.unsqueeze(-1)
            noisy_tokens = torch.where(selected, self.mask_idx, tokens)

        return noisy_tokens, mask_probability.unsqueeze(1).expand(-1, sequence_length)

    def _mop_embedding(self, token_embeddings, input_ids):
        """Fuse part-wise codebook embeddings with the learned MoP gate."""
        fallback = token_embeddings.mean(dim=-2).contiguous()
        body_start = self.id_dict["body"][0]
        lhand_start = self.id_dict["lhand"][0]
        rhand_start = self.id_dict["rhand"][0]
        valid = (
            (input_ids[..., 0] >= body_start)
            & (input_ids[..., 0] < body_start + self.body_codebook.shape[0])
            & (input_ids[..., 1] >= lhand_start)
            & (input_ids[..., 1] < lhand_start + self.lhand_codebook.shape[0])
            & (input_ids[..., 2] >= rhand_start)
            & (input_ids[..., 2] < rhand_start + self.rhand_codebook.shape[0])
        )
        if not valid.any():
            return fallback

        valid_ids = input_ids[valid]
        body_ids = valid_ids[..., 0] - body_start
        lhand_ids = valid_ids[..., 1] - lhand_start
        rhand_ids = valid_ids[..., 2] - rhand_start
        body_codes = self.body_codebook[body_ids]
        lhand_codes = self.lhand_codebook[lhand_ids]
        rhand_codes = self.rhand_codebook[rhand_ids]

        part_embeddings = torch.stack(
            [
                self.body_emb_mapper(body_codes),
                self.lhand_emb_mapper(lhand_codes),
                self.rhand_emb_mapper(rhand_codes),
            ],
            dim=-2,
        )
        gates = F.softmax(
            self.gate_net(torch.cat(
                [body_codes, lhand_codes, rhand_codes], dim=-1)),
            dim=-1,
        ).unsqueeze(-1)
        fused = (gates * part_embeddings).sum(dim=-2)
        fallback.view(-1, fallback.shape[-1])[valid.reshape(-1)] = fused
        return fallback

    def _get_logits(
        self,
        prompt_ids: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        token_embeddings = self.main_lm.embed_tokens
        prompt_length = 0 if prompt_ids is None else prompt_ids.shape[1]

        motion_attention_mask = torch.ones(
            (batch_size, input_ids.shape[1]), dtype=torch.long, device=device)
        if self.pretraining:
            combined_attention_mask = motion_attention_mask
        else:
            combined_attention_mask = torch.cat(
                [prompt_attention_mask, motion_attention_mask], dim=1)

        part_embeddings = token_embeddings(input_ids)
        if self.pretraining:
            motion_embeddings = part_embeddings.mean(dim=-2)
        else:
            motion_embeddings = self._mop_embedding(part_embeddings, input_ids)

        if prompt_ids is None:
            inputs_embeds = motion_embeddings
        else:
            inputs_embeds = torch.cat(
                [token_embeddings(prompt_ids), motion_embeddings], dim=1)

        outputs = self.main_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=_bidirectional_attention_mask(
                combined_attention_mask, inputs_embeds.dtype),
        )
        hidden_states = outputs.last_hidden_state[:, prompt_length:]
        logits = F.linear(hidden_states, token_embeddings.weight)
        logits = logits + self.final_logits_bias.to(device)

        if self.pretraining:
            logits = logits + self.mask_pretrain.to(device)
            return logits, logits, logits, hidden_states
        return (
            logits + self.mask_body.to(device),
            logits + self.mask_lhand.to(device),
            logits + self.mask_rhand.to(device),
            hidden_states,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor,
        labels_hand: torch.LongTensor,
        labels_rhand: torch.LongTensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        **_,
    ):
        device = self.main_lm.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device).masked_fill(labels.to(device) == -100, self.pad_idx)
        labels_hand = labels_hand.to(device).masked_fill(
            labels_hand.to(device) == -100, self.pad_idx)
        labels_rhand = labels_rhand.to(device).masked_fill(
            labels_rhand.to(device) == -100, self.pad_idx)
        text_length = input_ids.shape[1]

        if self.pretraining:
            text_inputs = torch.stack([input_ids, input_ids, input_ids], dim=-1)
            motion_inputs = torch.stack([labels, labels_hand, labels_rhand], dim=-1)
            clean_inputs = torch.cat([text_inputs, motion_inputs], dim=1)
            target_body = torch.cat([input_ids, labels], dim=1)
            target_lhand = torch.cat([input_ids, labels_hand], dim=1)
            target_rhand = torch.cat([input_ids, labels_rhand], dim=1)
            prompt_ids = None
        else:
            clean_inputs = torch.stack([labels, labels_hand, labels_rhand], dim=-1)
            target_body, target_lhand, target_rhand = labels, labels_hand, labels_rhand
            prompt_ids = input_ids

        noisy_inputs, mask_probability = self._add_noise(clean_inputs)
        mask_indices = noisy_inputs[..., 0] == self.mask_idx
        logits_body, logits_lhand, logits_rhand, hidden_states = self._get_logits(
            prompt_ids, noisy_inputs, attention_mask)

        if decoder_attention_mask is None:
            motion_valid = torch.ones_like(labels, dtype=torch.bool)
        else:
            motion_valid = decoder_attention_mask.to(device=device, dtype=torch.bool)
        valid_positions = (
            torch.cat([attention_mask.to(torch.bool), motion_valid], dim=1)
            if self.pretraining else motion_valid
        )
        position_ids = torch.arange(
            target_body.shape[1], device=device).unsqueeze(0)
        text_positions = position_ids < text_length if self.pretraining else None
        loss_function = CrossEntropyLoss(reduction="none")

        def masked_loss(logits, targets, part):
            selected_target_logits = logits.gather(
                -1, targets.unsqueeze(-1)).squeeze(-1)
            supported = torch.isfinite(selected_target_logits)
            code_lookup = torch.zeros(
                logits.shape[-1], dtype=torch.bool, device=device)
            code_lookup[torch.as_tensor(
                self.id_dict[part][:-3], dtype=torch.long, device=device)] = True
            valid_codes = code_lookup[targets]
            if self.pretraining and part == "body":
                valid_codes = valid_codes | text_positions
            elif self.pretraining:
                valid_codes = valid_codes & ~text_positions
            valid = valid_positions & supported & valid_codes
            selected = mask_indices & valid
            if not selected.any():
                finite = torch.where(
                    torch.isfinite(logits), logits, torch.zeros_like(logits))
                return finite.sum() * 0
            weighted = loss_function(
                logits[selected], targets[selected]
            ) / mask_probability[selected].clamp_min(1e-6)
            return weighted.sum() / valid.sum().clamp_min(1)

        return {
            "loss": masked_loss(logits_body, target_body, "body"),
            "loss_hand": masked_loss(logits_lhand, target_lhand, "lhand"),
            "loss_rhand": masked_loss(logits_rhand, target_rhand, "rhand"),
            "hidden_states": hidden_states,
            "text_len": text_length,
            "mask_indices": mask_indices,
        }

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_length=100, **_):
        """Generate all three sign streams with UTC enabled by construction."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        body = torch.full(
            (batch_size, max_length), self.mask_idx, dtype=torch.long, device=device)
        lhand = body.clone()
        rhand = body.clone()

        positions = torch.arange(max_length, device=device)
        quarter_grid = (positions % 4 == 0).expand(batch_size, -1)
        half_grid = (positions % 2 == 0).expand(batch_size, -1)
        full_grid = torch.ones_like(quarter_grid)
        timesteps = torch.linspace(1, 1e-5, DIFFUSION_STEPS + 1, device=device)

        for step in range(DIFFUSION_STEPS):
            # Reverse diffusion reaches the 0.75 and 0.5 noise pivots after
            # 25% and 50% of the denoising trajectory, respectively.
            if step <= int((1.0 - UTC_PIVOTS[0]) * DIFFUSION_STEPS):
                grid = quarter_grid
            elif step <= int((1.0 - UTC_PIVOTS[1]) * DIFFUSION_STEPS):
                grid = half_grid
            else:
                grid = full_grid

            masked = body == self.mask_idx
            logits_body, logits_lhand, logits_rhand, _ = self._get_logits(
                input_ids,
                torch.stack([body, lhand, rhand], dim=-1),
                attention_mask,
            )
            masked_body_logits = logits_body[masked]
            masked_lhand_logits = logits_lhand[masked]
            masked_rhand_logits = logits_rhand[masked]
            predictions_body = masked_body_logits.argmax(dim=-1)
            predictions_lhand = masked_lhand_logits.argmax(dim=-1)
            predictions_rhand = masked_rhand_logits.argmax(dim=-1)
            probabilities = F.softmax(masked_body_logits, dim=-1)
            confidence = probabilities.gather(
                -1, predictions_body.unsqueeze(-1)).squeeze(-1)
            confidence = confidence.masked_fill(~grid[masked], float("-inf"))
            confidence = confidence.reshape(batch_size, -1)

            masked_per_sample = masked.sum() // batch_size
            transfer_count = (
                int(masked_per_sample * (1 - timesteps[step + 1] / timesteps[step]))
                if step < DIFFUSION_STEPS - 1 else int(masked_per_sample)
            )
            if transfer_count == 0:
                continue
            transfer = torch.topk(confidence, transfer_count).indices
            transfer = (
                torch.arange(
                    0, masked.sum(), masked_per_sample,
                    dtype=torch.long, device=device).unsqueeze(-1)
                + transfer
            )

            next_body = torch.full_like(predictions_body, self.mask_idx)
            next_lhand = torch.full_like(predictions_lhand, self.mask_idx)
            next_rhand = torch.full_like(predictions_rhand, self.mask_idx)
            next_body[transfer] = predictions_body[transfer]
            next_lhand[transfer] = predictions_lhand[transfer]
            next_rhand[transfer] = predictions_rhand[transfer]
            body[masked] = next_body
            lhand[masked] = next_lhand
            rhand[masked] = next_rhand

        for sample_idx in range(batch_size):
            eos = (body[sample_idx] == self.eos_idx).nonzero(as_tuple=False)
            if eos.numel():
                end = int(eos[0, 0])
                body[sample_idx, end:] = self.eos_idx
                lhand[sample_idx, end:] = self.eos_idx
                rhand[sample_idx, end:] = self.eos_idx

        return {
            "outputs_re": body,
            "outputs_hand": lhand,
            "outputs_rhand": rhand,
        }
