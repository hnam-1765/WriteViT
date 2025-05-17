
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import Block, CrossBlock
from util.util import PositionalEncoding, PosCNN
from .blocks import Conv2dBlock, ResBlocks, ActFirstResBlock
from .Unifront import UnifontModule
from params import *

class Generator(nn.Module):

    def __init__(
        self,
        arg = None,
        embed_dim=256,
        depth=3,
        num_heads=4,
        mlp_ratio=4,
        drop=0.0,
        norm_layer=nn.LayerNorm,
        max_num_patch=100,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.layer_norm = None
        self.grid_size = None
        self.embed_dim = [256, 256,128, 128, 64, 32, 16]
        num_block = 4
        self.pos_enc = PositionalEncoding(embed_dim, drop, max_num_patch)
        self.query_embed = UnifontModule(
            embed_dim,
            ALPHABET,
            input_type="unifont",
            linear=True,
        )

        """Block 1"""
        index = 1
        self.blocks_2 = nn.ModuleList(
            [
                CrossBlock(
                    self.embed_dim[index],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth+2)
            ]
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim[index])
        self.tRGB_1 = nn.Sequential(
            nn.Conv2d(self.embed_dim[index], self.embed_dim[num_block], 3, 1, 1)
        )
        self.conv_1 =  self._make_upsample_block(self.embed_dim[index], self.embed_dim[index+1])

        """Block 2"""
        index+=1
        self.blocks_3 = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim[index],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.layer_norm3 = nn.LayerNorm(self.embed_dim[index])
        self.tRGB_2 = nn.Sequential(
            nn.Conv2d(self.embed_dim[index], self.embed_dim[num_block], 3, 1, 1)
        )
        self.conv_2 =  self._make_upsample_block(self.embed_dim[index], self.embed_dim[index+1])


        """Block 3"""
        index+=1
        self.blocks_4 = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim[index],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.layer_norm4 = nn.LayerNorm(self.embed_dim[index])
        self.tRGB_3 = nn.Sequential(
            nn.Conv2d(self.embed_dim[index], self.embed_dim[num_block], 3, 1, 1)
        )
        self.conv_3 =  self._make_upsample_block(self.embed_dim[index], self.embed_dim[index+1])


        """Block 4"""
        index+=1
        self.blocks_5 = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim[index],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )   
        self.layer_norm5 = nn.LayerNorm(self.embed_dim[index])

        self.pos_block = nn.ModuleList([PosCNN(i, i) for i in self.embed_dim])
        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        self.noise = torch.distributions.Normal(
            loc=torch.tensor([0.0]), scale=torch.tensor([1.0])
        )

        self.deconv = nn.Sequential(
            ResBlocks(
                2, self.embed_dim[index], norm="in", activation="relu", pad_type="reflect"
            ),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(
                self.embed_dim[index],
                self.embed_dim[index + 1],
                3,
                1,
                1,
                norm="in",
                activation="none",
                pad_type="reflect",
            ),
            Conv2dBlock(
                self.embed_dim[5],
                self.embed_dim[5],
                5,
                1,
                2,
                norm="in",
                activation="relu",
                pad_type="reflect",
            ),
            Conv2dBlock(
                self.embed_dim[5],
                1,
                7,
                1,
                3,
                norm="none",
                activation="tanh",
                pad_type="reflect",
            ),
        )
        self.initialize_weights()

    def _make_upsample_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(in_dim, out_dim, 3, 1, 1, norm="in", activation="none", pad_type="reflect"),
            Conv2dBlock(out_dim, out_dim, 3, 1, 1, norm="in", activation="relu", pad_type="reflect"),
        )

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _generate_features(self, src, tgt):
        b = src.size(0)
        start_h = 2
        start_w = tgt.size(1)

        src = src
        tmp = self.query_embed(tgt.clone())
        tgt = self.pos_enc(self.query_embed(tgt))

        stack_output = []
        for blk in self.blocks_2:
            tgt = blk(tgt, src)
            stack_output.append(tgt)
        h2 = stack_output[-1]

        tgt = torch.cat([h2, tmp], dim=1)
        tgt = self.layer_norm2(tgt)
        tgt = tgt.permute(0, 2, 1).view(b, self.embed_dim[1], start_h, start_w)
        x_1 = self.tRGB_1(tgt)

        tgt = self.conv_1(tgt)
        b, c, h, w = tgt.shape
        tgt = tgt.view(b, c, -1).permute(0, 2, 1)

        for j, blk in enumerate(self.blocks_3):
            tgt = blk(tgt)
            if j == 0:
                tgt = self.pos_block[2](tgt, h, w)
        tgt = self.layer_norm3(tgt).permute(0, 2, 1).view(b, self.embed_dim[2], h, w)
        x_2 = self.tRGB_2(tgt)

        tgt = self.conv_2(tgt)
        b, c, h, w = tgt.shape
        tgt = tgt.view(b, c, -1).permute(0, 2, 1)

        for j, blk in enumerate(self.blocks_4):
            tgt = blk(tgt)
            if j == 0:
                tgt = self.pos_block[3](tgt, h, w)
        tgt = self.layer_norm4(tgt).permute(0, 2, 1).view(b, self.embed_dim[3], h, w)
        x_3 = self.tRGB_3(tgt)

        tgt = self.conv_3(tgt)
        b, c, h, w = tgt.shape
        tgt = tgt.view(b, c, -1).permute(0, 2, 1)

        for j, blk in enumerate(self.blocks_5):
            tgt = blk(tgt)
            if j == 0:
                tgt = self.pos_block[4](tgt, h, w)
        tgt = self.layer_norm5(tgt).permute(0, 2, 1).view(b, self.embed_dim[4], h, w)

        fused = (
            F.interpolate(x_1, scale_factor=8)
            + F.interpolate(x_2, scale_factor=4)
            + F.interpolate(x_3, scale_factor=2)
            + tgt
        )
        noise = self.noise.sample(fused.size()).squeeze(-1).to(fused.device)
        return fused + noise

    def forward(self, src_w, tgt):
        features = self._generate_features(src_w, tgt)
        return self.deconv(features)

    def Eval(self, xw, QRS):
        outputs = []
        for i in range(QRS.shape[1]):
            tgt = QRS[:, i, :].squeeze(1)
            features = self._generate_features(xw, tgt)
            outputs.append(self.deconv(features).detach())
        return outputs