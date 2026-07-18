"""EMA vector-quantized motion tokenizer used by MaDiS."""

import torch
import torch.nn as nn
from torch import Tensor

from .tools.quantize_cnn import QuantizeEMAReset
from .tools.resnet import Resnet1D


class VQVae(nn.Module):
    """Temporal VQ-VAE matching the released body and hand checkpoints."""

    def __init__(
        self,
        nfeats: int,
        quantizer: str = "ema_reset",
        code_num=512,
        code_dim=512,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        norm=None,
        activation="relu",
    ):
        super().__init__()
        if quantizer != "ema_reset":
            raise ValueError("MaDiS uses the EMA-reset vector quantizer")
        self.code_num = code_num
        self.code_dim = code_dim
        self.nfeats = nfeats
        self.stride_t = stride_t
        self.down_t = down_t
        self.encoder = Encoder(
            nfeats,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.decoder = Decoder(
            nfeats,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)

    @staticmethod
    def preprocess(features):
        return features.transpose(1, 2)

    @staticmethod
    def postprocess(features):
        return features.transpose(1, 2)

    def forward(self, features: Tensor, return_latent=False, **_):
        encoded = self.encoder(self.preprocess(features))
        quantized, loss, perplexity, _ = self.quantizer(encoded)
        output = self.postprocess(self.decoder(quantized))
        if return_latent:
            return output, loss, perplexity, self.postprocess(quantized)
        return output, loss, perplexity

    def encode(self, features: Tensor):
        encoded = self.encoder(self.preprocess(features))
        flattened = self.quantizer.preprocess(encoded)
        indices = self.quantizer.quantize(flattened)
        return indices.view(features.shape[0], -1), None

    def decode(self, indices: Tensor):
        single_sample = indices.ndim == 1
        if single_sample:
            indices = indices.unsqueeze(0)
        quantized = self.quantizer.dequantize(indices).transpose(1, 2).contiguous()
        output = self.postprocess(self.decoder(quantized))
        return output


class Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks = [
            nn.Conv1d(input_emb_width, width, 3, 1, 1),
            nn.ReLU(),
        ]
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    activation=activation,
                    norm=norm,
                ),
            ))
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs, **_):
        return self.model(inputs)


class Decoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = [
            nn.Conv1d(output_emb_width, width, 3, 1, 1),
            nn.ReLU(),
        ]
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=True,
                    activation=activation,
                    norm=norm,
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(width, width, 3, 1, 1),
            ))
        blocks.extend([
            nn.Conv1d(width, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(width, input_emb_width, 3, 1, 1),
        ])
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs, **_):
        return self.model(inputs)
