"""Language/token interface for the MaDiS model."""

import os
import pickle
import random
import time
from typing import List, Optional

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer

from mGPT.archs.masked_diffusion import MaskedDiffusionLanguageModel


class MaDiSLanguageModel(nn.Module):
    """Qwen3 tokenizer adapter plus the paper's masked diffusion model."""

    def __init__(
        self,
        model_path: str,
        stage: str,
        motion_codebook_size: int,
        hand_codebook_size: int,
        rhand_codebook_size: int,
        body_codebook: Tensor,
        hand_codebook: Tensor,
        rhand_codebook: Tensor,
        max_length: int = 256,
        framerate: float = 20.0,
        down_t: int = 4,
    ):
        super().__init__()
        self.stage = stage
        self.m_codebook_size = motion_codebook_size
        self.hand_codebook_size = hand_codebook_size
        self.rhand_codebook_size = rhand_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        self.tokenizer.add_tokens(["<mask>"], special_tokens=True)
        motion_tokens = [
            f"<motion_id_{index}>"
            for index in range(motion_codebook_size + 3)
        ]
        hand_tokens = [
            f"<hand_id_{index}>"
            for index in range(hand_codebook_size + 3)
        ]
        rhand_tokens = [
            f"<rhand_id_{index}>"
            for index in range(rhand_codebook_size + 3)
        ]
        self.tokenizer.add_tokens(motion_tokens + hand_tokens + rhand_tokens)

        with open(os.path.join(model_path, "map_ids.pkl"), "rb") as handle:
            self.tok_id_to_emb_id = pickle.load(handle)
        next_id = len(self.tok_id_to_emb_id)
        for token in ["<mask>", *motion_tokens, *hand_tokens, *rhand_tokens]:
            tokenizer_id = self.tokenizer.convert_tokens_to_ids(token)
            self.tok_id_to_emb_id[tokenizer_id] = next_id
            next_id += 1
        self.emb_id_to_tok_id = {
            embedding_id: tokenizer_id
            for tokenizer_id, embedding_id in self.tok_id_to_emb_id.items()
        }

        motion_tokenizer_ids = self.tokenizer.convert_tokens_to_ids(motion_tokens)
        hand_tokenizer_ids = self.tokenizer.convert_tokens_to_ids(hand_tokens)
        rhand_tokenizer_ids = self.tokenizer.convert_tokens_to_ids(rhand_tokens)
        motion_ids = [self.tok_id_to_emb_id[index] for index in motion_tokenizer_ids]
        hand_ids = [self.tok_id_to_emb_id[index] for index in hand_tokenizer_ids]
        rhand_ids = [self.tok_id_to_emb_id[index] for index in rhand_tokenizer_ids]

        special_ids = {
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        }
        vocabulary_ids = set(self.tokenizer.get_vocab().values())

        def removed_ids(allowed_ids):
            removed = vocabulary_ids - set(allowed_ids) - special_ids
            return [
                self.tok_id_to_emb_id[index]
                for index in removed
                if index in self.tok_id_to_emb_id
            ]

        self.pad_idx = self.tok_id_to_emb_id[self.tokenizer.pad_token_id]
        self.eos_idx = self.tok_id_to_emb_id[self.tokenizer.eos_token_id]
        self.mask_idx = self.tok_id_to_emb_id[
            self.tokenizer.convert_tokens_to_ids("<mask>")]

        self.language_model = MaskedDiffusionLanguageModel(
            model_path=model_path,
            len_token=len(self.tok_id_to_emb_id),
            ids_remove_motion=removed_ids(motion_tokenizer_ids),
            ids_remove_hand=removed_ids(hand_tokenizer_ids),
            ids_remove_rhand=removed_ids(rhand_tokenizer_ids),
            ids_motion=motion_ids,
            ids_hand=hand_ids,
            ids_rhand=rhand_ids,
            pad_idx=self.pad_idx,
            eos_idx=self.eos_idx,
            mask_idx=self.mask_idx,
            body_codebook=body_codebook,
            hand_codebook=hand_codebook,
            rhand_codebook=rhand_codebook,
            stage=stage,
        )

    def _map_ids(self, input_ids: Tensor, to_embeddings: bool):
        mapping = self.tok_id_to_emb_id if to_embeddings else self.emb_id_to_tok_id
        fallback = self.pad_idx if to_embeddings else self.tokenizer.pad_token_id
        flat = input_ids.reshape(-1)
        mapped = [mapping.get(int(value), fallback) for value in flat]
        input_ids.copy_(torch.tensor(
            mapped, dtype=input_ids.dtype, device=input_ids.device
        ).reshape_as(input_ids))

    def _tokenize(self, strings, labels: bool, device):
        encoded = self.tokenizer(
            strings,
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device=device, dtype=torch.long)
        attention_mask = encoded.attention_mask.to(device=device, dtype=torch.long)
        self._map_ids(input_ids, to_embeddings=True)
        if labels:
            input_ids = input_ids.masked_fill(attention_mask == 0, -100)
        return input_ids, attention_mask

    @staticmethod
    def _token_strings(tokens: Tensor, lengths: List[int], prefix: str):
        strings = []
        for sample, length in zip(tokens, lengths):
            values = sample.detach().cpu().tolist()[:length]
            strings.append("".join(
                f"<{prefix}_id_{int(value)}>" for value in values))
        return strings

    @staticmethod
    def _fill_template(template, length, motion_string, text):
        return (
            template.replace("<Caption_Placeholder>", text)
            .replace("<Motion_Placeholder>", motion_string)
            .replace("<Frame_Placeholder>", str(length))
        )

    def _templates(self, tasks, lengths, motion_strings, texts, prefix):
        inputs, outputs = [], []
        for task, length, motion_string, text in zip(
                tasks, lengths, motion_strings, texts):
            inputs.append(self._fill_template(
                random.choice(task["input"]), length, motion_string, text))
            output_template = random.choice(task["output"])
            outputs.append(self._fill_template(
                output_template, length, motion_string, text).replace(
                    "<Motion_Placeholder>", motion_string))
        return inputs, outputs

    def forward(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks,
        src: List[str],
        name: List[str],
    ):
        del src, name
        body_strings = self._token_strings(
            motion_tokens[..., 0], lengths, "motion")
        lhand_strings = self._token_strings(
            motion_tokens[..., 1], lengths, "hand")
        rhand_strings = self._token_strings(
            motion_tokens[..., 2], lengths, "rhand")

        inputs, body_outputs = self._templates(
            tasks, lengths, body_strings, texts, "motion")
        _, lhand_outputs = self._templates(
            tasks, lengths, lhand_strings, texts, "hand")
        _, rhand_outputs = self._templates(
            tasks, lengths, rhand_strings, texts, "rhand")

        device = motion_tokens.device
        source_ids, source_mask = self._tokenize(inputs, labels=False, device=device)
        body_ids, motion_mask = self._tokenize(
            body_outputs, labels=True, device=device)
        lhand_ids, _ = self._tokenize(
            lhand_outputs, labels=True, device=device)
        rhand_ids, _ = self._tokenize(
            rhand_outputs, labels=True, device=device)
        return self.language_model(
            input_ids=source_ids,
            attention_mask=source_mask,
            labels=body_ids,
            labels_hand=lhand_ids,
            labels_rhand=rhand_ids,
            decoder_attention_mask=motion_mask,
        )

    def _decode_stream(self, embedding_ids: Tensor, prefix: str):
        tokenizer_ids = embedding_ids.clone()
        self._map_ids(tokenizer_ids, to_embeddings=False)
        strings = self.tokenizer.batch_decode(
            tokenizer_ids, skip_special_tokens=True)
        decoded = []
        for string in strings:
            compact = "".join(string.split())
            pieces = compact.replace("><", ">|<").split("|")
            values = []
            for piece in pieces:
                marker = f"<{prefix}_id_"
                if piece.startswith(marker) and piece.endswith(">"):
                    try:
                        values.append(int(piece[len(marker):-1]))
                    except ValueError:
                        pass
            if not values:
                values = [0]
            decoded.append(torch.tensor(
                values, dtype=torch.long, device=embedding_ids.device))
        return decoded

    def generate_direct(self, texts: List[str], max_length: int, src=None, **_):
        del src
        device = self.language_model.main_lm.device
        source_ids, source_mask = self._tokenize(
            texts, labels=False, device=device)
        start = time.perf_counter()
        outputs = self.language_model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=max_length,
        )
        elapsed = time.perf_counter() - start
        return {
            "outputs_tokens": self._decode_stream(
                outputs["outputs_re"], "motion"),
            "outputs_tokens_hand": self._decode_stream(
                outputs["outputs_hand"], "hand"),
            "outputs_tokens_rhand": self._decode_stream(
                outputs["outputs_rhand"], "rhand"),
            "time_cost": elapsed,
        }

    def generate_conditional(
        self,
        texts: List[str],
        lengths: Optional[List[int]] = None,
        tasks=None,
        src: Optional[List[str]] = None,
        max_length: int = 100,
        **_,
    ):
        if texts is None:
            raise ValueError("Text prompts are required for sign generation")
        if lengths is None:
            lengths = [0] * len(texts)
        if tasks is None:
            tasks = [{
                "input": ["<Caption_Placeholder>"],
                "output": ["<Motion_Placeholder>"],
            }] * len(texts)
        empty_motion = [""] * len(texts)
        inputs, _ = self._templates(
            tasks, lengths, empty_motion, texts, "motion")
        return self.generate_direct(
            inputs, max_length=max_length, src=src)
