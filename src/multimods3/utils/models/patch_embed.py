# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

class PatchEmbed(torch.nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ConvEmbed15D(torch.nn.Module):

    def __init__(self, *, patch_size: int, embed_dim: int, in_chans=0, stride: int | None = None):
        """
        Conv-embed ``S`` sensors of ``T`` samples each. Assume each sensor as a 1D signal that is to be kept separate
        from the others (thus ``1.5D``); can be used for patch embedding (see ``stride`` explanation).

        * Given ``B×C×T×S`` input (``B``: batch size, ``C``: ``in_chans>0``), produce ``B×t×S×E`` output
          (``E``: ``embed_dim``, ``t := floor((T - patch_size) / stride + 1)``).
        * Given ``B×T×S`` input and ``in_chans=0``, extend input to ``B×1×T×S`` and proceed as above.

        :param patch_size: patch size along the temporal dimension
        :param embed_dim: embedding dimension (= number of convolution kernels)
        :param in_chans: number of channels per sample
        :param stride: stride along ``T`` dimension; if None (default), use ``patch_size`` as stride, effectively
            resulting in a patch embedding along ``T``
        :return: ``B×t×S×E`` output (see above)
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        k = (patch_size, 1)
        s = (patch_size if stride is None else stride, 1)
        self.proj = torch.nn.Conv2d(in_channels=max(1, in_chans), out_channels=embed_dim, kernel_size=k, stride=s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1) if self.in_chans == 0 else x
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchEmbed3D(torch.nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.proj = torch.nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        # B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
