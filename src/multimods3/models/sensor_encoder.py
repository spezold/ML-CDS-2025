from dataclasses import dataclass
import math

from torch import nn
import torch

from multimods3.models.vision_transformer import ViTArgs, VideoArgs, ViTSizeArgs
from multimods3.utils.models.modules import WeightSharedCrossAttentionBlock, MLP, WeightSharedCrossAttention
from multimods3.utils.models.patch_embed import ConvEmbed15D
from multimods3.utils.models.pos_embs import get_2d_sincos_pos_embed


def _patching_remainder(len_data: int, len_patch: int, len_stride: int | None = None) -> int:
    # What remains from patch embedding, using the `ConvND` equation from PyTorch docs
    return int((len_data - len_patch) / (len_patch if len_stride is None else len_stride) + 1)


def _patching_remainder_video(args: VideoArgs):
    # What remains from `PatchEmbed3D` in ViT, using the `ConvND` equation from PyTorch docs
    r_temporal = _patching_remainder(len_data=args.num_frames, len_patch=args.tubelet_size)
    r_spatial = _patching_remainder(len_data=args.img_size, len_patch=args.patch_size)
    return r_temporal * r_spatial * r_spatial


@dataclass(frozen=True, kw_only=True)
class SensorsArgs:
    num_sensors: int
    num_samples: int
    frame_step: int

    @classmethod
    def derive_from(cls, *, num_sensors: int, frame_step: int, num_frames: int) -> "SensorsArgs":
        return cls(num_sensors=num_sensors, num_samples=(num_frames - 1) * frame_step + 1, frame_step=frame_step)


class SensorsEncoder(nn.Module):

    def __init__(
            self,
            size_args: ViTSizeArgs,
            num_frames: int,
            frame_step: int,
            video_patch_size: int,
            sensors_args: SensorsArgs,
            vit_args: ViTArgs | None = None
    ):
        super().__init__()
        stride = 1  # Use a convolutional embedding along temporal dimension
        vit_args = ViTArgs() if vit_args is None else vit_args  # Use default arguments
        num_frames_in_clip = 1 + (num_frames - 1) * frame_step
        num_tokens_samples = _patching_remainder(len_data=num_frames_in_clip,
                                                 len_patch=video_patch_size,
                                                 len_stride=stride)
        num_tokens_sensors = sensors_args.num_sensors
        self.init_std = vit_args.init_std

        self._embed = ConvEmbed15D(patch_size=video_patch_size, embed_dim=size_args.embed_dim, stride=stride)
        pos = get_2d_sincos_pos_embed(grid_size=(num_tokens_samples, num_tokens_sensors), embed_dim=size_args.embed_dim)
        self._positional_embedding = nn.Parameter(torch.from_numpy(pos).unsqueeze_(0), requires_grad=False)

        # Shared layers → shared weights
        to_q = nn.Linear(size_args.embed_dim, size_args.embed_dim, bias=vit_args.qkv_bias)
        to_kv = nn.Linear(size_args.embed_dim, 2 * size_args.embed_dim, bias=vit_args.qkv_bias)
        proj = nn.Linear(size_args.embed_dim, size_args.embed_dim, bias=True)

        self._blocks = nn.ModuleList([
            WeightSharedCrossAttentionBlock(
                num_heads=size_args.num_heads,
                to_q=to_q,
                to_kv=to_kv,
                proj=proj,
                mlp_ratio=size_args.mlp_ratio,
                act_layer=nn.GELU,
                norm_layer=vit_args.norm_layer
            ) for _ in range(size_args.depth)])
        self._norm = vit_args.norm_layer(size_args.embed_dim)  # FIXME: Currently not used

        self.apply(self._init_weights)
        self._rescale_blocks()

    @staticmethod
    def _zero_init_linear(m: nn.Module):
        assert isinstance(m, nn.Linear)
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)

    def _init_weights(self, m):  # From VisionTransformer, but ensure initial 0 delta for residuals (MLP, xattn)
        if isinstance(m, MLP):
            for name, child in m.named_children():
                if name == "fc2":
                    self._zero_init_linear(child)
                else:
                    child.apply(self._init_weights)
        if isinstance(m, WeightSharedCrossAttention):
            for name, child in m.named_children():
                if name == "proj":
                    self._zero_init_linear(child)
                else:
                    child.apply(self._init_weights)
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

    def _rescale_blocks(self):  # From VisionTransformer
        def rescale(param, layer_id_):
            param.div_(math.sqrt(2.0 * layer_id_))

        for layer_id, layer in enumerate(self._blocks):
            rescale(layer.xattn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, emb: torch.Tensor, sens: list[torch.Tensor]):
        # We have:
        #   - shape of `emb` (video or previous encodings): b×(c·t_v·s·s)×d
        #   - shape of `sens` (sensor data): c×b×t_s,in×n
        #   - b: batch size
        #   - c: number of clips/segments per sample
        #   - d: embedding dimension
        #   - n: number of sensors
        #   - s: number of tokens along (each) spatial axis of the video frames
        #   - t_s,in: number of sensor values along temporal axis before conv-embedding
        #   - t_s,em: number of sensor values after conv-embedding
        #   - t_v: number of video tokens along temporal axis
        b, c = len(emb), len(sens)
        sens = torch.stack(sens).flatten(0, 1)  # c×b×t_s,in×n → (c·b)×t_s,in×n
        sens = self._embed(sens).flatten(1, 2)  # (c·b)×t_s,in×n → (c·b)×t_s,em×n×d → (c·b)×(t_s,em·n)×d
        sens += self._positional_embedding
        sens = sens.view(c, b, *sens.shape[1:])  # (c·b)×(t_s,em·n)×d → c×b×(t_s,em·n)×d
        sens = sens.transpose(0, 1).flatten(1, 2)  # c×b×(t_s,em·n)×d → b×c×(t_s,em·n)×d → b×(c·t_s,em·n)×d

        for blk in self._blocks:
            emb = blk(emb, sens)

        return emb
