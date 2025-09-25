# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math
from functools import partial

import torch
import torch.nn as nn

from multimods3.utils.models.patch_embed import PatchEmbed, PatchEmbed3D
from multimods3.utils.models.modules import Block
from multimods3.utils.models.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from multimods3.utils.masks import apply_masks


@dataclass(frozen=True, kw_only=True)
class ViTSizeArgs:
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float


class ViTSize(Enum):
    VIT_TINY = ViTSizeArgs(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4)
    VIT_SMALL = ViTSizeArgs(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4)
    VIT_BASE = ViTSizeArgs(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4)
    VIT_LARGE = ViTSizeArgs(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4)
    VIT_HUGE = ViTSizeArgs(embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4)
    VIT_GIANT = ViTSizeArgs(embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48 / 11)
    VIT_GIGANTIC = ViTSizeArgs(embed_dim=1664, depth=48, num_heads=16, mlp_ratio=64 / 13)


@dataclass(frozen=True, kw_only=True)
class VideoArgs:
    img_size: int
    patch_size: int
    num_frames: int
    frame_step: int
    tubelet_size: int
    uniform_power: bool


@dataclass(frozen=True, kw_only=True)
class ViTArgs:
    in_chans: int = 3
    qkv_bias: bool = True
    qk_scale: float | None = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6)
    init_std: float = 0.02
    out_layers: Sequence[int] | None = None


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, size: ViTSize | ViTSizeArgs, video_args: VideoArgs, vit_args: ViTArgs | None = None):
        super().__init__()
        size_args = size if isinstance(size, ViTSizeArgs) else size.value
        vit_args = ViTArgs() if vit_args is None else vit_args  # Use default arguments
        self.num_features = self.embed_dim = size_args.embed_dim
        self.num_heads = size_args.num_heads
        self.out_layers = vit_args.out_layers

        self.input_size = video_args.img_size
        self.patch_size = video_args.patch_size

        self.num_frames = video_args.num_frames
        self.tubelet_size = video_args.tubelet_size
        self.is_video = video_args.num_frames > 1

        # Tokenize pixels with convolution
        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=video_args.patch_size,
                tubelet_size=video_args.tubelet_size,
                in_chans=vit_args.in_chans,
                embed_dim=size_args.embed_dim)
            self.num_patches = (
                (video_args.num_frames // video_args.tubelet_size)
                * (video_args.img_size // video_args.patch_size)
                * (video_args.img_size // video_args.patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(
                patch_size=video_args.patch_size,
                in_chans=vit_args.in_chans,
                embed_dim=size_args.embed_dim)
            self.num_patches = (
                (video_args.img_size // video_args.patch_size)
                * (video_args.img_size // video_args.patch_size)
            )

        # Position embedding
        self.uniform_power = video_args.uniform_power
        self.pos_embed = None
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, size_args.embed_dim),
            requires_grad=False)

        # Attention Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=size_args.embed_dim,
                num_heads=size_args.num_heads,
                mlp_ratio=size_args.mlp_ratio,
                qkv_bias=vit_args.qkv_bias,
                qk_scale=vit_args.qk_scale,
                drop=vit_args.drop_rate,
                act_layer=nn.GELU,
                attn_drop=vit_args.attn_drop_rate,
                norm_layer=vit_args.norm_layer)
            for _ in range(size_args.depth)])
        self.norm = vit_args.norm_layer(size_args.embed_dim)

        # ------ initialize weights
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)  # sincos pos-embed
        self.init_std = vit_args.init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim,
                grid_size,
                grid_depth,
                cls_token=False,
                uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def forward(self, x, masks=None):
        """
        :param x: input image/video
        :param masks: indices of patch tokens to mask (remove)
        """

        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # Tokenize input
        pos_embed = self.pos_embed
        if pos_embed is not None:
            pos_embed = self.interpolate_pos_encoding(x, pos_embed)
        x = self.patch_embed(x)
        if pos_embed is not None:
            x += pos_embed

        # Mask away unwanted tokens (if masks provided)
        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        # Fwd prop
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask=masks)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))

        if self.out_layers is not None:
            return outs

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):

        _, N, dim = pos_embed.shape

        if self.is_video:

            # If pos_embed already correct size, just return
            _, _, T, H, W = x.shape
            if H == self.input_size and W == self.input_size and T == self.num_frames:
                return pos_embed

            # Convert depth, height, width of input to be measured in patches
            # instead of pixels/frames
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            # Compute the initialized shape of the positional embedding measured
            # in patches
            N_t = self.num_frames // self.tubelet_size
            N_h = N_w = self.input_size // self.patch_size
            assert N_h * N_w * N_t == N, 'Positional embedding initialized incorrectly'

            # Compute scale factor for spatio-temporal interpolation
            scale_factor = (T/N_t, H/N_h, W/N_w)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode='trilinear')
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        else:

            # If pos_embed already correct size, just return
            _, _, H, W = x.shape
            if H == self.input_size and W == self.input_size:
                return pos_embed

            # Compute scale factor for spatial interpolation
            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode='bicubic')
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed
