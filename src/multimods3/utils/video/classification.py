# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from dataclasses import dataclass
from logging import getLogger

import torch
import torch.nn as nn

from multimods3.utils.models.pos_embs import get_1d_sincos_pos_embed
from multimods3.utils.masks import apply_masks
from multimods3.utils.video.transforms import EvalVideoTransform, VideoTransform, VideoTransformArgs


class ClipAggregation(nn.Module):
    """
    Process each clip independently and concatenate all tokens
    """

    def __init__(
        self,
        model,
        tubelet_size,
        max_frames,
        use_pos_embed,
        attend_across_segments
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.attend_across_segments = attend_across_segments
        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_T, embed_dim),
                requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):

        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, T, H, W = x[0][0].size()

        # Concatenate all spatial and temporal views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)
        outputs = self.model(x)
        _, N, D = outputs.size()

        T = T // self.tubelet_size  # Num temporal tokens
        N = N // T  # Num spatial tokens

        # Unroll outputs into a 2D array [spatial_views x temporal_views]
        eff_B = B * num_views_per_clip
        all_outputs = [[] for _ in range(num_views_per_clip)]
        for i in range(num_clips):
            o = outputs[i*eff_B:(i+1)*eff_B]
            for j in range(num_views_per_clip):
                all_outputs[j].append(o[j*B:(j+1)*B])

        if not self.attend_across_segments:
            return all_outputs

        for i, outputs in enumerate(all_outputs):

            # Concatenate along temporal dimension
            outputs = [o.reshape(B, T, N, D) for o in outputs]
            outputs = torch.cat(outputs, dim=1).flatten(1, 2)

            # Compute positional embedding
            if (self.pos_embed is not None) and (clip_indices is not None):
                clip_indices = [c[:, ::self.tubelet_size] for c in clip_indices]
                pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, F, D]
                pos_embed = apply_masks(pos_embed, clip_indices, concat=False)  # list(Tensor([B, T, D]))
                pos_embed = torch.cat(pos_embed, dim=1)  # concatenate along temporal dimension
                pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, N, 1)  # [B, T*num_clips, N, D]
                pos_embed = pos_embed.flatten(1, 2)
                outputs += pos_embed

            all_outputs[i] = outputs

        return all_outputs


@dataclass(frozen=True, kw_only=True)
class VideoTransformsArgs:
    num_views_per_clip: int
    video_transform_args: VideoTransformArgs


def video_transforms_from(args: VideoTransformsArgs) -> EvalVideoTransform | VideoTransform:
    vargs = args.video_transform_args
    if not vargs.train and args.num_views_per_clip > 1:
        getLogger().info("Make EvalVideoTransform, multi-view")
        _frames_augmentation = EvalVideoTransform(
            num_views_per_clip=args.num_views_per_clip,
            short_side_size=vargs.crop_size,
            normalize=vargs.normalize,
        )
    else:
        _frames_augmentation = VideoTransform(vargs)
    return _frames_augmentation


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
