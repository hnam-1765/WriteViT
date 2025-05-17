import torch
import torch.nn as nn
import torch.nn.functional as F
from params import *
from .Attention import Block
from util.util import PosCNN, PositionalEncoding
from .Resnet18 import ResNet18


class LayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)
    

class ViT_OCR(nn.Module):

    def __init__(
        self,
        nb_cls=VOCAB_SIZE,
        embed_dim=256,
        depth=3,
        num_heads=8,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
        qkv_bias=True,
        spectral=True,
        max_num_patch=100,
        drop=0.0,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.layer_norm = LayerNorm()
        self.patch_embed = ResNet18(embed_dim)
        self.embed_dim = embed_dim
        self.pos_block = PosCNN(embed_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim = self.embed_dim,
                    num_heads = num_heads,
                    mlp_ratio = mlp_ratio,
                    qkv_bias = qkv_bias,
                    norm_layer = norm_layer,
                    spectral = spectral,
                )
                for i in range(depth)
            ]
        )
        self.pos_enc = PositionalEncoding(embed_dim, drop, max_num_patch)
        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        self.head = torch.nn.Linear(embed_dim, nb_cls)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x = self.layer_norm(x)
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)

        for j, blk in enumerate(self.blocks):
            x = blk(x)
            if j == 0:
                x = self.pos_block(x, h, w)  # PEG here

        x = self.norm(x)
        feature = x
        x = self.head(x)
        x = self.layer_norm(x)

        return feature, x
