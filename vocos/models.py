"""Vocos backbones. Currently support two models:
    - VocosBackbone:       ConvNeXt-based
    - VocosResNetBackbone: ResBlock1-based
"""

from torch import nn, Tensor
from torch.nn.utils import weight_norm

from vocos.modules import ConvNeXtBlock, ResBlock1, AdaLayerNorm
from vocos.domain import UnitSeries, FeatSeries


class Backbone(nn.Module):
    """ABC of the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: UnitSeries, **kwargs) -> FeatSeries:
        """
        Args:
            x - Unit series
        Returns:
              - Feature series
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """Vocos backbone, PreNet-PreNorm-N*ConvNeXt-PostNorm.
    
    Supports additional conditioning with Adaptive Layer Normalization.
    """

    def __init__(
        self,
        input_channels:   int,
        dim:              int,
        intermediate_dim: int,
        num_layers:       int,
        layer_scale_init_value: None | float = None,
        adanorm_num_embeddings: None | int   = None,
        kernel_prenet:                 int   = 7,
        kernel_convnx:                 int   = 7,
        c:                             bool  = False,
        learnable_norm:                bool  = True,
    ):
        """
        Args:
            input_channels         - Number of input features channels
            dim                    - Hidden dimension of the model
            intermediate_dim       - Feature dimension size of ConvNeXtBlock Intermediate layer
            num_layers             - Number of ConvNeXtBlock layers
            layer_scale_init_value - Initial value for layer scaling. Defaults to `1 / num_layers`.
            adanorm_num_embeddings - Number of embeddings for AdaLayerNorm. None means non-conditional model
        """
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        # Validation
        if (adanorm_num_embeddings) and (not learnable_norm):
            raise RuntimeError(f"Not supported argument combination: adanorm_num_embeddings {adanorm_num_embeddings} & learnable_norm {learnable_norm}")

        feat_i, feat_o = input_channels, dim

        self.embed = nn.Conv1d(feat_i, feat_o, kernel_prenet, padding="same")
        self.adanorm = adanorm_num_embeddings is not None
        self.norm = AdaLayerNorm(adanorm_num_embeddings, feat_o, eps=1e-6) if adanorm_num_embeddings else nn.LayerNorm(feat_o, eps=1e-6, elementwise_affine=learnable_norm)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(dim=feat_o, intermediate_dim=intermediate_dim, kernel=kernel_convnx, layer_scale_init_value=layer_scale_init_value, adanorm_num_embeddings=adanorm_num_embeddings, c=c, learnable_norm=learnable_norm)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(feat_o, eps=1e-6, elementwise_affine=learnable_norm)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: UnitSeries, bandwidth_id: None | Tensor = None) -> FeatSeries:
        """
        Args:
            x :: (B, Feat=i, Frame=frm) - Unit series
        Returns:
              :: (B, Frame=frm, Feat=o) - Feature series
        """

        # PreNet :: (B, Feat=i, Frame) -> (B, Feat=o, Frame)
        x = self.embed(x)

        # PreNorm :: (B, Feat, Frame) -> (B, Frame, Feat) -> (B, Frame, Feat) -> (B, Feat, Frame)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # MainNet :: (B, Feat=o, Frame) -> (B, Feat=o, Frame) - ConvNeXt layers
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)

        # PostNorm :: (B, Feat=o, Frame) -> (B, Frame, Feat=o)
        x = self.final_layer_norm(x.transpose(1, 2))

        return x


class VocosResNetBackbone(Backbone):
    """Vocos ResNet backbone, PreNet-N*ResBlock."""

    def __init__(self,
        input_channels:         int,
        dim:                    int,
        num_blocks:             int,
        layer_scale_init_value: None | float = None,
    ):
        """
        Args:
            input_channels         - Number of input features channels
            dim                    - Hidden dimension of the model
            num_blocks             - Number of ResBlock1 blocks.
            layer_scale_init_value - Initial value for layer scaling
        """

        super().__init__()

        feat_i, feat_o = input_channels, dim

        self.embed = weight_norm(nn.Conv1d(feat_i, feat_o, kernel_size=3, padding="same"))
        # MainNet - N * ResBlock_k3, not MRF
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(*[ResBlock1(dim=feat_o, layer_scale_init_value=layer_scale_init_value) for _ in range(num_blocks)])

    def forward(self, x: UnitSeries, **kwargs) -> FeatSeries:
        """
        Args:
            x :: (B, Feat=i, Frame=frm) - Unit series
        Returns:
              :: (B, Frame=frm, Feat=o) - Feature series
        """

        #                     PreNet                MainNet              Transpose
        # :: (B, Feat=i, Frame) -> (B, Feat=o, Frame) -> (B, Feat=o, Frame) -> (B, Frame, Feat=o)
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)

        return x
