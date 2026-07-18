"""EMA-reset vector quantizer used by the MaDiS sign tokenizer."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu=0.99):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.initialized = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer("codebook", torch.zeros(nb_code, code_dim))

    def _tile(self, values):
        if len(values) >= self.nb_code:
            return values
        repeats = math.ceil(self.nb_code / len(values))
        noise = torch.randn_like(values.repeat(repeats, 1)) * (
            0.01 / math.sqrt(values.shape[1]))
        return values.repeat(repeats, 1) + noise

    def _initialize(self, values):
        self.codebook.copy_(self._tile(values)[:self.nb_code])
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(
            self.nb_code, device=values.device, dtype=values.dtype)
        self.initialized = True

    @staticmethod
    def preprocess(values):
        return values.transpose(1, 2).contiguous().view(-1, values.shape[1])

    def quantize(self, values):
        codebook = self.codebook.t()
        distance = (
            values.square().sum(-1, keepdim=True)
            - 2 * values @ codebook
            + codebook.square().sum(0, keepdim=True)
        )
        return distance.argmin(-1)

    def dequantize(self, indices):
        return F.embedding(indices, self.codebook)

    @torch.no_grad()
    def _statistics(self, indices):
        counts = torch.bincount(indices, minlength=self.nb_code).to(self.codebook)
        probability = counts / counts.sum().clamp_min(1)
        perplexity = torch.exp(
            -(probability * (probability + 1e-7).log()).sum())
        return counts, perplexity

    @torch.no_grad()
    def _update(self, values, indices):
        counts, perplexity = self._statistics(indices)
        sums = torch.zeros_like(self.codebook)
        sums.index_add_(0, indices, values)
        random_codes = self._tile(values)[:self.nb_code]
        self.code_sum.mul_(self.mu).add_(sums, alpha=1 - self.mu)
        self.code_count.mul_(self.mu).add_(counts, alpha=1 - self.mu)
        used = (self.code_count >= 1).unsqueeze(1)
        updated = self.code_sum / self.code_count.unsqueeze(1).clamp_min(1e-7)
        self.codebook.copy_(torch.where(used, updated, random_codes))
        return perplexity

    def forward(self, values):
        batch_size, _, frames = values.shape
        flattened = self.preprocess(values)
        if self.training and not self.initialized:
            self._initialize(flattened)
        indices = self.quantize(flattened)
        quantized = self.dequantize(indices)
        if self.training:
            perplexity = self._update(flattened.detach(), indices)
        else:
            _, perplexity = self._statistics(indices)
        loss = F.mse_loss(flattened, quantized.detach())
        quantized = flattened + (quantized - flattened).detach()
        quantized = quantized.view(batch_size, frames, -1).transpose(1, 2)
        return quantized.contiguous(), loss, perplexity, indices
