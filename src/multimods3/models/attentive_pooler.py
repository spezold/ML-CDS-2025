# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from multimods3.utils.models.modules import Block, CrossAttention, CrossAttentionBlock


@dataclass(frozen=True, kw_only=True)
class AttentivePoolerArgs:
    embed_dim: int
    num_heads: int
    num_queries: int = 1
    mlp_ratio: float = 4.0
    depth: int = 1
    norm_layer: nn.Module = nn.LayerNorm
    init_std: float = 0.02
    qkv_bias: bool = True
    complete_block: bool = True


class AttentivePooler(nn.Module):
    """ Attentive Pooler """
    def __init__(self, args: AttentivePoolerArgs):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, args.num_queries, args.embed_dim))

        self.complete_block = args.complete_block
        if args.complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=args.embed_dim,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                qkv_bias=args.qkv_bias,
                norm_layer=args.norm_layer)
        else:
            self.cross_attention_block = CrossAttention(
                dim=args.embed_dim,
                num_heads=args.num_heads,
                qkv_bias=args.qkv_bias)

        self.blocks = None
        if args.depth > 1:
            self.blocks = nn.ModuleList([
                Block(
                    dim=args.embed_dim,
                    num_heads=args.num_heads,
                    mlp_ratio=args.mlp_ratio,
                    qkv_bias=args.qkv_bias,
                    qk_scale=False,
                    norm_layer=args.norm_layer)
                for i in range(args.depth - 1)])

        self.init_std = args.init_std
        nn.init.trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

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

    def forward(self, x):
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)
        return q


@dataclass(frozen=True, kw_only=True)
class AttentiveClassifierArgs:
    num_classes: int
    pooler_args: AttentivePoolerArgs


class AttentiveClassifier(nn.Module):
    """ Attentive Classifier """
    def __init__(self, args: AttentiveClassifierArgs):
        super().__init__()
        self.pooler = AttentivePooler(args.pooler_args)
        self.linear = nn.Linear(args.pooler_args.embed_dim, args.num_classes, bias=True)

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.linear(x)
        return x
