# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Self-contained model building blocks ported from Makani's
# `models.common` package so the LNO model has no Makani dependency at
# import or run time. Single-rank only (no model parallelism).

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import amp
from torch.utils.checkpoint import checkpoint


# -----------------------------------------------------------------------------
# DropPath (timm-style stochastic depth)
# -----------------------------------------------------------------------------

@torch.compile(fullgraph=False)
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample, used in residual blocks."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# -----------------------------------------------------------------------------
# LayerScale (CaiT-style per-channel scale)
# -----------------------------------------------------------------------------

class LayerScale(nn.Module):
    def __init__(self, num_chans: int = 3, init_value: float = 0.1):
        super().__init__()
        self.num_chans = num_chans
        self.weight = nn.Parameter(torch.full((num_chans, 1, 1, 1), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(x, self.weight, groups=self.num_chans)


# -----------------------------------------------------------------------------
# 2-layer MLP (NCHW or "traditional" linear)
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        output_bias: bool = True,
        input_format: str = "nchw",
        drop_rate: float = 0.0,
        drop_type: str = "iid",
        checkpointing: bool = False,
        gain: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if input_format == "nchw":
            fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
        elif input_format == "traditional":
            fc1 = nn.Linear(in_features, hidden_features, bias=True)
        else:
            raise NotImplementedError(f"Unsupported input_format='{input_format}'")

        nn.init.normal_(fc1.weight, mean=0.0, std=math.sqrt(2.0 / in_features))
        nn.init.constant_(fc1.bias, 0.0)

        act = act_layer()

        if input_format == "traditional" and drop_type == "features":
            raise NotImplementedError(
                "input_format='traditional' is incompatible with drop_type='features'"
            )

        if input_format == "nchw":
            fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
        else:
            fc2 = nn.Linear(hidden_features, out_features, bias=output_bias)

        nn.init.normal_(fc2.weight, mean=0.0, std=math.sqrt(gain / hidden_features))
        if fc2.bias is not None:
            nn.init.constant_(fc2.bias, 0.0)

        if drop_rate > 0.0:
            if drop_type == "iid":
                drop = nn.Dropout(drop_rate)
            elif drop_type == "features":
                drop = nn.Dropout2d(drop_rate)
            else:
                raise NotImplementedError(f"Unsupported drop_type='{drop_type}'")
        else:
            drop = nn.Identity()

        self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)

    @torch.compiler.disable(recursive=False)
    def _checkpoint_forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self.fwd, x, use_reentrant=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._checkpoint_forward(x) if self.checkpointing else self.fwd(x)


# -----------------------------------------------------------------------------
# EncoderDecoder (stack of 1x1 convs / linears + final projection)
# -----------------------------------------------------------------------------

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        act_layer,
        gain: float = 1.0,
        input_format: str = "nchw",
        groups: int = 1,
    ):
        super().__init__()
        modules = []
        current_dim = input_dim
        for _ in range(num_layers):
            if input_format == "nchw":
                modules.append(nn.Conv2d(current_dim, hidden_dim, 1, bias=True, groups=groups))
            elif input_format == "traditional":
                modules.append(nn.Linear(current_dim, hidden_dim, bias=True))
            else:
                raise NotImplementedError(f"Unsupported input_format='{input_format}'")

            fan_in = (current_dim // groups) if input_format == "nchw" else current_dim
            nn.init.normal_(modules[-1].weight, mean=0.0, std=math.sqrt(2.0 / fan_in))
            if modules[-1].bias is not None:
                nn.init.constant_(modules[-1].bias, 0.0)
            modules.append(act_layer())
            current_dim = hidden_dim

        if input_format == "nchw":
            modules.append(nn.Conv2d(current_dim, output_dim, 1, bias=False, groups=groups))
        else:
            modules.append(nn.Linear(current_dim, output_dim, bias=False))

        fan_in = (current_dim // groups) if input_format == "nchw" else current_dim
        nn.init.normal_(modules[-1].weight, mean=0.0, std=math.sqrt(gain / fan_in))
        if modules[-1].bias is not None:
            nn.init.constant_(modules[-1].bias, 0.0)

        self.fwd = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fwd(x)


# -----------------------------------------------------------------------------
# Spectral convolution (SFNO-style)
# -----------------------------------------------------------------------------

@torch.compile
def _contract_lmwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bgixy,gioxy->bgoxy", ac, bc)


@torch.compile
def _contract_lwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bgixy,giox->bgoxy", ac, bc)


@torch.compile
def _contract_sep_lmwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bgixy,gixy->bgixy", ac, bc)


@torch.compile
def _contract_sep_lwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bgixy,gix->bgixy", ac, bc)


def _contract_dense_pytorch(
    x: torch.Tensor, weight: torch.Tensor,
    separable: bool = False, operator_type: str = "diagonal", complex: bool = True,
) -> torch.Tensor:
    """Dense spectral-conv contraction (complex-only path; real path unused here)."""
    x = x.contiguous()
    if not complex:
        raise NotImplementedError("real-tensor spectral contraction not used by LNO")
    if separable:
        if operator_type == "diagonal":
            x = _contract_sep_lmwise(x, weight)
        elif operator_type == "dhconv":
            x = _contract_sep_lwise(x, weight)
        else:
            raise ValueError(f"Unknown operator_type='{operator_type}'")
    else:
        if operator_type == "diagonal":
            x = _contract_lmwise(x, weight)
        elif operator_type == "dhconv":
            x = _contract_lwise(x, weight)
        else:
            raise ValueError(f"Unknown operator_type='{operator_type}'")
    return x.contiguous()


class SpectralConv(nn.Module):
    """SFNO-style spectral convolution via SHT.

    Single-rank only (no model parallelism). Faithful port of Makani's
    ``models.common.spectral_convolution.SpectralConv`` minus the
    ``DistributedInverseRealSHT`` branch.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels: int,
        out_channels: int,
        num_groups: int = 1,
        operator_type: str = "dhconv",
        separable: bool = False,
        bias: bool = False,
        gain: float = 1.0,
    ):
        super().__init__()
        if in_channels % num_groups != 0:
            raise ValueError("in_channels must be divisible by num_groups")
        if out_channels % num_groups != 0:
            raise ValueError("out_channels must be divisible by num_groups")

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.operator_type = operator_type
        self.separable = separable

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
            or self.forward_transform.nlon != self.inverse_transform.nlon
        )
        if hasattr(self.forward_transform, "grid"):
            self.scale_residual = self.scale_residual or (
                self.forward_transform.grid != self.inverse_transform.grid
            )

        weight_shape = [num_groups, in_channels // num_groups]
        if not separable:
            weight_shape += [out_channels // num_groups]

        if operator_type == "diagonal":
            weight_shape += [self.modes_lat, self.modes_lon]
        elif operator_type == "dhconv":
            weight_shape += [self.modes_lat]
        else:
            raise ValueError(f"Unsupported operator_type='{operator_type}'")

        scale = math.sqrt(gain / (in_channels // num_groups)) * torch.ones(self.modes_lat, dtype=torch.complex64)
        # First (l=0) mode is real; counter the implicit factor of 2.
        scale[0] *= math.sqrt(2.0)
        init = scale * torch.randn(*weight_shape, dtype=torch.complex64)
        self.weight = nn.Parameter(init)

        self._contract = partial(
            _contract_dense_pytorch,
            separable=separable, complex=True, operator_type=operator_type,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        residual = x

        with amp.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.float32)
            x = self.forward_transform(x).contiguous()
            if self.scale_residual:
                residual = self.inverse_transform(x)

        if self.scale_residual:
            residual = residual.to(dtype=dtype)

        B, C, H, W = x.shape
        x = x.reshape(B, self.num_groups, C // self.num_groups, H, W)
        xp = self._contract(x, self.weight)
        x = xp.reshape(B, self.out_channels, H, W).contiguous()

        with amp.autocast(device_type=x.device.type, enabled=False):
            x = self.inverse_transform(x)

        x = x.to(dtype=dtype)
        if hasattr(self, "bias"):
            x = x + self.bias
        return x, residual
