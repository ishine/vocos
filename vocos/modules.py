"""Modules used several modules."""

import torch
from torch import nn, Tensor
from torch.nn.utils import weight_norm, remove_weight_norm
from extorch import Conv1dEx                               # pyright: ignore [reportMissingTypeStubs]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block, Res[DwConv-Norm-PwConv-GELU-PwConv[-γ]].

    Adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.
    """

    def __init__(self,
        dim:                    int,
        intermediate_dim:       int,
        kernel:                 int,
        layer_scale_init_value: None | float = None,
        adanorm_num_embeddings: None | int   = None,
        c:                      bool         = False,
    ):
        """
        Args:
            dim                    - Number of input channels.
            intermediate_dim       - Dimensionality of the intermediate layer.
            layer_scale_init_value - Initial value for the layer scale. None means no scaling.
            adanorm_num_embeddings - Number of embeddings for AdaLayerNorm. None means non-conditional LayerNorm.
        """
        super().__init__()

        feat_io, feat_h = dim, intermediate_dim
        # DepthwiseConv/Norm
        self.dwconv     = nn.Conv1d(feat_io, feat_io, kernel,              padding="same", groups=feat_io)
        if c:
            self.dwconv =  Conv1dEx(feat_io, feat_io, kernel, causal=True, padding="same", groups=feat_io)
        self.adanorm = adanorm_num_embeddings is not None
        self.norm = AdaLayerNorm(adanorm_num_embeddings, feat_io, eps=1e-6) if adanorm_num_embeddings else nn.LayerNorm(feat_io, eps=1e-6)
        # PointwiseConv/GELU/PointwiseConv/γ
        self.pwconv1 = nn.Linear(feat_io, feat_h)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(feat_h, feat_io)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(feat_io), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x: Tensor, cond_embedding_id: None | Tensor = None) -> Tensor:
        """
        Args:
            x                 :: (B, Feat=io, Frame=frm) - Input  feature series
            cond_embedding_id
        Returns:
                              :: (B, Feat=io, Frame=frm) - Output feature series
        """

        residual = x

        # Depthwise :: (B, Feat=io, Frame) -> (B, Feat=io, Frame)
        x = self.dwconv(x)

        # Norm :: (B, Feat=io, Frame) -> (B, Frame, Feat=io) -> (B, Frame, Feat=io)
        x = x.transpose(1, 2)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)

        # Pointwise :: (B, Frame, Feat=io) -> (B, Frame, Feat=h) -> (B, Frame, Feat=io) -> (B, Feat=io, Frame)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x if (self.gamma is not None) else x
        x = x.transpose(1, 2)

        # Residual
        x = residual + x

        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale + shift
        return x


class ResBlock1(nn.Module):
    """
    ResBlock adapted from HiFi-GAN V1 (https://github.com/jik876/hifi-gan) with dilated 1D convolutions,
    but without upsampling layers.
    """

    def __init__(self,
        dim:                    int,
        layer_scale_init_value: None | float = None,
    ):
        """
        Args:
            dim                    - Number of input channels.
            layer_scale_init_value - Initial value for the layer scale. None means no scaling.
        """
        super().__init__()

        kernel_size = 3
        dilation    = (1, 3, 5)
        lrelu_slope = 0.1

        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(dim, dim, kernel_size, dilation=dilation[0], padding=self.get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(dim, dim, kernel_size, dilation=dilation[1], padding=self.get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(dim, dim, kernel_size, dilation=dilation[2], padding=self.get_padding(kernel_size, dilation[2]))),
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(dim, dim, kernel_size,                       padding="same")),
            weight_norm(nn.Conv1d(dim, dim, kernel_size,                       padding="same")),
            weight_norm(nn.Conv1d(dim, dim, kernel_size,                       padding="same")),
        ])

        self.gamma = nn.ParameterList([
            nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True) if layer_scale_init_value is not None else None,
            nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True) if layer_scale_init_value is not None else None,
            nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True) if layer_scale_init_value is not None else None,
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
            # Res[LReLU-DilConv-LReLU-Conv[-γ]]
            xt = torch.nn.functional.leaky_relu(x,  negative_slope=self.lrelu_slope)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, negative_slope=self.lrelu_slope)
            xt = c2(xt)
            if gamma is not None:
                xt = gamma * xt
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

    @staticmethod
    def get_padding(kernel_size: int, dilation: int) -> int:
        return int((kernel_size * dilation - dilation) / 2)


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)
