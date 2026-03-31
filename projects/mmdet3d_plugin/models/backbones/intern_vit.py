import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import BACKBONES


class LayerNorm(nn.Module):
    """LayerNorm over channel dim for NCHW tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def get_norm(norm: Optional[str], out_channels: int) -> Optional[nn.Module]:
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        if norm == "BN":
            return nn.BatchNorm2d(out_channels)
        if norm == "SyncBN":
            return nn.SyncBatchNorm(out_channels)
        if norm == "GN":
            return nn.GroupNorm(32, out_channels)
        if norm == "LN":
            return LayerNorm(out_channels)
        raise KeyError(f"Unsupported norm='{norm}'. Supported: '', BN, SyncBN, GN, LN.")
    return norm(out_channels)


class Conv2d(nn.Conv2d):
    """Conv2d wrapper supporting (optional) norm + activation."""

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SimpleFeaturePyramid(nn.Module):
    """SimpleFeaturePyramid (ViTDet-style) for a single input feature map."""

    def __init__(
        self,
        scale_factors=(4, 2, 1, 0.5),
        in_channels: int = 1024,
        out_channels: int = 256,
        out_indices=(2, 3, 4, 5),
        norm: str = "LN",
    ):
        super().__init__()
        self.scale_factors = list(scale_factors)
        strides = [int(16 / scale) for scale in self.scale_factors]
        self.output_strides: List[int] = []

        self.stages = nn.ModuleList()
        use_bias = norm == ""
        for idx, scale in enumerate(self.scale_factors):
            out_dim = in_channels
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                    get_norm(norm, in_channels // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2),
                ]
                out_dim = in_channels // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)]
                out_dim = in_channels // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)]
            elif scale == 0.25:
                layers = [nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4)]
            else:
                raise ValueError(f"Unsupported scale factor: {scale}")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )

            stage = int(math.log2(strides[idx]))
            if stage in set(out_indices):
                self.stages.append(nn.Sequential(*layers))
                self.output_strides.append(int(strides[idx]))

    def forward(self, x: torch.Tensor):
        return [stage(x) for stage in self.stages]


def _maybe_get_attr(obj: Any, names) -> Optional[Any]:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _get_patch_size_from_config(cfg: Any) -> int:
    # InternViT configs can vary; cover common cases.
    for holder in [cfg, _maybe_get_attr(cfg, ["vision_config"]), _maybe_get_attr(cfg, ["model_config"])]:
        if holder is None:
            continue
        ps = _maybe_get_attr(holder, ["patch_size"])
        if ps is not None:
            if isinstance(ps, (tuple, list)):
                return int(ps[0])
            return int(ps)
    return 14  # reasonable default for InternViT; will be validated at runtime


@BACKBONES.register_module()
class InternViT(nn.Module):
    """
    HuggingFace InternViT backbone wrapper.

    It returns a list of feature maps (NCHW). If `sim_fpn` is provided, it builds
    multi-scale features using a ViTDet-style SimpleFeaturePyramid; otherwise it
    returns a single feature map.
    """

    def __init__(
        self,
        model_name_or_path: str = "OpenGVLab/InternViT-300M-448px-V2_5",
        pretrained: bool = True,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
        sim_fpn: Optional[Dict[str, Any]] = None,
        frozen: bool = False,
        hf_kwargs: Optional[Dict[str, Any]] = None,
        align_mlvl_to_input: bool = True,
        single_scale_stride: int = 16,
    ):
        super().__init__()

        try:
            from transformers import AutoConfig, AutoModel  # type: ignore
        except Exception as e:
            raise ImportError(
                "InternViT backbone requires HuggingFace transformers. "
                'Please install it, e.g. `pip install "transformers>=4.38"`. '
                f"Original import error: {e}"
            )

        self.frozen = frozen
        self.patch_size = None
        self.adapter = None
        self.align_mlvl_to_input = align_mlvl_to_input
        self.single_scale_stride = int(single_scale_stride)

        hf_kwargs = dict(hf_kwargs or {})
        if revision is not None:
            hf_kwargs.setdefault("revision", revision)
        hf_kwargs.setdefault("trust_remote_code", trust_remote_code)

        if pretrained:
            self.vit = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
        else:
            cfg = AutoConfig.from_pretrained(model_name_or_path, **hf_kwargs)
            self.vit = AutoModel.from_config(cfg, trust_remote_code=trust_remote_code)

        self.patch_size = _get_patch_size_from_config(getattr(self.vit, "config", None))

        # infer embed dim
        hidden_size = _maybe_get_attr(getattr(self.vit, "config", None), ["hidden_size", "embed_dim", "dim"])
        if hidden_size is None:
            hidden_size = getattr(self.vit, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Cannot infer InternViT hidden size from config.")
        self.embed_dim = int(hidden_size)

        if sim_fpn is not None:
            cfg = dict(sim_fpn)
            cfg.setdefault("in_channels", self.embed_dim)
            self.adapter = SimpleFeaturePyramid(**cfg)
            self.fpn_strides: List[int] = list(self.adapter.output_strides)
        else:
            self.fpn_strides = [self.single_scale_stride]

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def _forward_hf(self, x: torch.Tensor):
        # Support both common signatures: (pixel_values=...) and (x=...)/positional.
        try:
            return self.vit(pixel_values=x, return_dict=True)
        except TypeError:
            return self.vit(x, return_dict=True)

    def _align_mlvl_feats(self, feats: List[torch.Tensor], img_h: int, img_w: int) -> List[torch.Tensor]:
        """Resize pyramid levels to (img // stride) to match PETR FocalHead heatmap grid (pad_shape // 16)."""
        if not self.align_mlvl_to_input:
            return feats
        if len(feats) != len(self.fpn_strides):
            raise RuntimeError(
                f"len(feats)={len(feats)} != len(fpn_strides)={len(self.fpn_strides)}. "
                "Check sim_fpn out_indices vs adapter outputs."
            )
        out: List[torch.Tensor] = []
        for feat, s in zip(feats, self.fpn_strides):
            th = max(img_h // int(s), 1)
            tw = max(img_w // int(s), 1)
            if feat.shape[-2:] != (th, tw):
                feat = F.interpolate(feat, size=(th, tw), mode="bilinear", align_corners=False)
            out.append(feat)
        return out

    def forward(self, x: torch.Tensor, img_metas=None):
        out = self._forward_hf(x)

        if hasattr(out, "last_hidden_state"):
            tokens = out.last_hidden_state  # (B, N, C)
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            tokens = out[0]
        else:
            raise RuntimeError("Unexpected InternViT output format; cannot extract token features.")

        b, n, c = tokens.shape
        img_h, img_w = int(x.shape[-2]), int(x.shape[-1])
        ps = int(self.patch_size)
        h_expect, w_expect = img_h // ps, img_w // ps

        if n == h_expect * w_expect + 1:
            tokens = tokens[:, 1:, :]
            n = h_expect * w_expect

        if n == h_expect * w_expect:
            h, w = h_expect, w_expect
        else:
            # Fallback: square grid (legacy HF layouts); final align still fixes FPN vs image.
            side = int(math.sqrt(n))
            if side * side != n:
                raise RuntimeError(
                    f"Cannot reshape tokens to feature map. token_len={n}, "
                    f"expected {h_expect*w_expect} (+ optional cls) from input {img_h}x{img_w} / patch {ps}, "
                    f"or a perfect square."
                )
            h = w = side

        feat = tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

        if self.adapter is not None:
            feats = self.adapter(feat)
            return self._align_mlvl_feats(feats, img_h, img_w)
        feats = [feat]
        return self._align_mlvl_feats(feats, img_h, img_w)

