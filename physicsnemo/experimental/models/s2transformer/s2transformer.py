# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Spherical transformer (S2Transformer) for gridded geophysical fields.

This module provides :class:`PrecipAttentionNet`, a fully attention-based
encoder / processor / decoder backbone that operates directly on the sphere.
Every internal mixer is a spherical transformer block built around
torch-harmonics' :class:`~torch_harmonics.AttentionS2` /
:class:`~torch_harmonics.NeighborhoodAttentionS2` (Bonev et al., NeurIPS 2025,
https://arxiv.org/abs/2505.11157):

.. code-block:: text

    AttentionEncoder (in_channels -> embed_dim,  H_in,W_in -> h,w)
        + optional positional embedding on the (h, w) internal grid
        -> [SphericalTransformerBlock] x num_layers
    AttentionDecoder (embed_dim -> out_channels, h,w -> H_out,W_out)
        + optional big-skip residual from the input

Each :class:`SphericalTransformerBlock` is the canonical pre-norm transformer:
``x = x + drop_path(self_attn(norm0(x)))`` followed by
``x = x + drop_path(mlp(norm1(x)))``. The attention is quadrature-aware on the
sphere (softmax weights include the geodesic Jacobian), so the layer remains
SO(3)-equivariant in expectation and is well-behaved at the poles.

``torch_harmonics`` is an optional dependency; it is only imported when a model
from this module is instantiated (see :func:`physicsnemo.core.version_check`).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.core.version_check import (
    OptionalImport,
    check_version_spec,
    register_package_hint,
)

from .layers import MLP, DropPath

# torch_harmonics provides the spherical attention / resampling / SHT kernels.
# It is optional: the imports below are lazy (``th.<Attr>`` triggers the import
# on first access), so ``import physicsnemo`` never fails for users who do not
# need this model. A clear, actionable error is raised at model construction
# time when the package is missing (see ``PrecipAttentionNet.__init__``).
register_package_hint(
    "torch_harmonics",
    "torch_harmonics is required for the spherical transformer (S2Transformer) "
    "models.\nInstall with:\n  pip install torch_harmonics>=0.7.0\n"
    "  pip install nvidia-physicsnemo[harmonics]",
)
TORCH_HARMONICS_AVAILABLE = check_version_spec("torch_harmonics", hard_fail=False)
th = OptionalImport("torch_harmonics")


def _neighborhood_radius_rad(nlat: int, factor: float) -> float:
    """Geodesic-cap radius (radians) for neighborhood attention.

    Expressed in units of latitude spacing on an equiangular-style grid:
    ``radius = factor * pi / (nlat - 1)``. ``factor = 1.0`` therefore covers
    one latitude ring, ``factor = 4.0`` four rings, etc. Attention has no kernel
    basis (unlike a DISCO convolution), so the radius is just a geodesic angle.

    Parameters
    ----------
    nlat : int
        Number of latitude points of the grid the attention runs on.
    factor : float
        Neighborhood radius in units of latitude rings.

    Returns
    -------
    float
        Cutoff radius in radians.
    """
    return float(factor) * math.pi / float(max(int(nlat) - 1, 1))


# -----------------------------------------------------------------------------
# Attention encoder / decoder
# -----------------------------------------------------------------------------


class AttentionEncoder(nn.Module):
    r"""Spherical-attention encoder: ``(in_chans, H_in, W_in) -> (out_chans, h, w)``.

    Pipeline:

    1. Pointwise 1x1 lift ``in_chans -> out_chans`` on the input grid.
    2. :class:`~torch_harmonics.NeighborhoodAttentionS2` **cross-resolution
       downsampling**: the attention itself aggregates from the full-resolution
       K/V tokens to the coarse Q positions, so no explicit resampling module is
       needed. Q must be at ``out_shape`` and K/V at ``inp_shape``. Q is seeded
       by a spherically-correct bilinear downsample (:class:`~torch_harmonics.ResampleS2`)
       which also serves as the residual base; an optional positional embedding
       is added to Q before the attention.

    When ``inp_shape == out_shape`` there is no downsampling to do, so the
    resampling and the cross-resolution attention are skipped entirely (not even
    constructed) and the encoder degenerates to the pointwise 1x1 lift.

    Requires ``nlon_in % nlon_out == 0`` (a hard constraint of the underlying
    spherical attention kernel).

    Parameters
    ----------
    inp_shape : tuple of int
        Input grid shape :math:`(H_{in}, W_{in})`.
    out_shape : tuple of int
        Output (latent) grid shape :math:`(h, w)`.
    in_chans : int
        Number of input channels.
    out_chans : int
        Number of output channels.
    grid_in : str
        Quadrature grid type of the input grid.
    grid_out : str
        Quadrature grid type of the latent grid.
    num_heads : int
        Number of attention heads.
    theta_cutoff_factor : float
        Neighborhood radius (in latitude rings) for the cross-resolution attention.
    qk_norm : bool
        Whether to apply RMSNorm to the query/key projections.
    bias : bool
        Whether the linear projections use a bias.
    attn_optimized_kernel : bool
        Whether to use the optimized CUDA neighborhood-attention kernel.
    attn_dim : int, optional, default=None
        Channel width at which the (expensive, full-resolution) cross-attention
        runs. ``None`` means no bottleneck (``out_chans``).
    pos_embed_max_degree : int, optional, default=None
        Maximum spherical-harmonic degree of the owned positional embedding
        (only used when the encoder bottlenecks).
    """

    def __init__(
        self,
        inp_shape: Tuple[int, int],
        out_shape: Tuple[int, int],
        in_chans: int,
        out_chans: int,
        grid_in: str,
        grid_out: str,
        num_heads: int,
        theta_cutoff_factor: float,
        qk_norm: bool,
        bias: bool,
        attn_optimized_kernel: bool,
        attn_dim: Optional[int] = None,
        pos_embed_max_degree: Optional[int] = None,
    ):
        super().__init__()
        self.out_chans = int(out_chans)
        # The cross-resolution attention (and its bilinear-downsample residual
        # base) only exist when the encoder actually changes resolution. When
        # inp_shape == out_shape there is nothing to downsample, so the encoder
        # degenerates to the pointwise lift.
        self._needs_resample = inp_shape != out_shape

        # Channel width at which the (expensive, full-resolution) cross-attention
        # runs. ``None`` -> out_chans (no bottleneck). When set smaller, the lift
        # maps straight to attn_dim, the attention runs narrow against the
        # full-resolution K/V, and a cheap 1x1 conv at the LATENT grid expands
        # attn_dim -> out_chans, slashing full-res attention memory by
        # ~ (out_chans / attn_dim)x.
        self.attn_dim = (
            int(attn_dim) if (attn_dim and self._needs_resample) else self.out_chans
        )
        if self._needs_resample and self.attn_dim % num_heads != 0:
            raise ValueError(
                f"AttentionEncoder: attn_dim ({self.attn_dim}) must be divisible "
                f"by num_heads ({num_heads})."
            )

        # When bottlenecking, lift straight to the narrow attn_dim; otherwise the
        # lift already produces the full out_chans.
        lift_out = self.attn_dim if self._needs_resample else self.out_chans
        self.lift = nn.Conv2d(in_chans, lift_out, kernel_size=1, bias=bias)
        nn.init.normal_(self.lift.weight, std=math.sqrt(2.0 / max(1, in_chans)))
        if bias:
            nn.init.zeros_(self.lift.bias)

        # Spatial PE applied to Q only. When bottlenecking, the shared embed_dim
        # PE no longer matches the narrow attention width, so the encoder owns a
        # PE at attn_dim; otherwise it is assigned externally by the parent model.
        self._owns_pe = self._needs_resample and (self.attn_dim != self.out_chans)
        if self._owns_pe:
            self.pos_embed: Optional[nn.Module] = _SpectralPositionEmbedding(
                out_shape,
                self.attn_dim,
                grid_out,
                max_degree=pos_embed_max_degree,
            )
        else:
            self.pos_embed = None

        if self._needs_resample:
            # Spherically-correct bilinear downsample to the latent grid.
            # Used both as the residual base and as Q seed for the cross-attention.
            self.resample = th.ResampleS2(
                inp_shape[0],
                inp_shape[1],
                out_shape[0],
                out_shape[1],
                grid_in=grid_in,
                grid_out=grid_out,
            )
            # Cross-resolution attention: K/V at inp_shape, Q at out_shape.
            # Output is added as a residual on top of the bilinear downsample.
            theta_cutoff = _neighborhood_radius_rad(inp_shape[0], theta_cutoff_factor)
            self.attn = th.NeighborhoodAttentionS2(
                in_channels=self.attn_dim,
                in_shape=inp_shape,
                out_shape=out_shape,
                grid_in=grid_in,
                grid_out=grid_out,
                num_heads=num_heads,
                theta_cutoff=theta_cutoff,
                use_qknorm=qk_norm,
                bias=bias,
                out_channels=self.attn_dim,
                optimized_kernel=attn_optimized_kernel,
            )

        # Expand attn_dim -> out_chans at the latent grid (cheap, low-res). When
        # not bottlenecking this is an Identity.
        if self._needs_resample and self.attn_dim != self.out_chans:
            self.expand = nn.Conv2d(self.attn_dim, self.out_chans, kernel_size=1, bias=bias)
            nn.init.normal_(self.expand.weight, std=math.sqrt(2.0 / max(1, self.attn_dim)))
            if bias:
                nn.init.zeros_(self.expand.bias)
        else:
            self.expand = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` of shape :math:`(B, C_{in}, H_{in}, W_{in})` to :math:`(B, C_{out}, h, w)`."""
        x = self.lift(x)  # (B, attn_dim | out_chans, H, W)
        if not self._needs_resample:
            # Same shape: pointwise lift only, no resampling/attention.
            return x
        dtype = x.dtype
        with amp.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            # Residual base: spherically-correct bilinear downsample.
            q_base = self.resample(x_f)  # (B, attn_dim, h, w)
            # Q for attention: add spatial PE so attention is position-aware.
            q_attn = self.pos_embed(q_base) if self.pos_embed is not None else q_base
            # Cross-attention correction: what the full-resolution input adds
            # beyond the bilinear base.
            delta = self.attn(q_attn, key=x_f, value=x_f)  # (B, attn_dim, h, w)
            x = (q_base + delta).to(dtype)  # residual
        x = self.expand(x)  # (B, out_chans, h, w)
        return x


class AttentionDecoder(nn.Module):
    r"""Spherical-attention decoder: ``(in_chans, h, w) -> (out_chans, H_out, W_out)``.

    Mirror of :class:`AttentionEncoder`. When the latent and output grids differ,
    the decoder upsamples with a spherically-correct bilinear
    :class:`~torch_harmonics.ResampleS2` (or an SHT round-trip when
    ``upsample_sht=True``) and adds a learned cross-resolution "attnup"
    correction on top of that base:

    .. code-block:: text

        kv   = reduce(z)                          # optional attn_dim bottleneck (latent grid)
        base = upsample(kv)                       # -> output grid (residual base)
        q    = base + pos_embed                   # position-aware query on the output grid
        x    = base + attn(q, key=kv, value=kv)   # cross-resolution upsampling attention
        out  = project(x)                         # attn_dim -> out_chans

    Only the query is at the output resolution; K/V stay on the cheap latent grid.
    When the latent grid already equals the output grid (no resolution change) the
    upsample, positional embedding and attention are skipped entirely and the
    decoder degenerates to the pointwise channel projection (``reduce`` ->
    ``project``) -- exactly mirroring the encoder, whose attention only runs when
    resampling.

    The cross-resolution upsampling attention requires ``nlon_out % nlon_in == 0``.
    ``theta_cutoff_factor`` is a geodesic cap on the OUTPUT grid; the latent K/V
    spacing is ~``H_out / h`` output rings, so pick a factor at least as large as
    the up-sampling ratio (~``scale_factor``) so every fine query reaches >= 1
    coarse point.

    Parameters
    ----------
    inp_shape : tuple of int
        Input (latent) grid shape :math:`(h, w)`.
    out_shape : tuple of int
        Output grid shape :math:`(H_{out}, W_{out})`.
    in_chans : int
        Number of input channels.
    out_chans : int
        Number of output channels.
    grid_in : str
        Quadrature grid type of the latent grid.
    grid_out : str
        Quadrature grid type of the output grid.
    num_heads : int
        Number of attention heads.
    theta_cutoff_factor : float
        Neighborhood radius (in output-grid latitude rings) for the upsampling
        attention. Use a value at least as large as the up-sampling ratio.
    qk_norm : bool
        Whether to apply RMSNorm to the query/key projections.
    bias : bool
        Whether the linear projections use a bias.
    attn_optimized_kernel : bool
        Whether to use the optimized CUDA neighborhood-attention kernel.
    upsample_sht : bool, optional, default=False
        Whether to upsample with an SHT round-trip instead of :class:`~torch_harmonics.ResampleS2`.
    attn_dim : int, optional, default=None
        Channel width at which the (full-resolution) attention query runs.
        ``None`` means no bottleneck (``in_chans``).
    pos_embed_max_degree : int, optional, default=None
        Maximum spherical-harmonic degree for the query positional embedding.
    """

    def __init__(
        self,
        inp_shape: Tuple[int, int],
        out_shape: Tuple[int, int],
        in_chans: int,
        out_chans: int,
        grid_in: str,
        grid_out: str,
        num_heads: int,
        theta_cutoff_factor: float,
        qk_norm: bool,
        bias: bool,
        attn_optimized_kernel: bool,
        upsample_sht: bool = False,
        attn_dim: Optional[int] = None,
        pos_embed_max_degree: Optional[int] = None,
    ):
        super().__init__()
        self.in_chans = int(in_chans)
        self._needs_resample = inp_shape != out_shape
        # Channel width at which the (full-resolution) attention query runs.
        # ``None`` -> in_chans (no bottleneck). When set smaller, a cheap 1x1 conv
        # at the LATENT grid reduces in_chans -> attn_dim before the attnup, and
        # the final projection maps attn_dim -> out_chans.
        self.attn_dim = int(attn_dim) if attn_dim else self.in_chans
        if self._needs_resample and self.attn_dim % num_heads != 0:
            raise ValueError(
                f"AttentionDecoder: attn_dim ({self.attn_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        # Optional channel bottleneck at the cheap latent grid (Identity when off).
        if self.attn_dim != self.in_chans:
            self.reduce = nn.Conv2d(self.in_chans, self.attn_dim, kernel_size=1, bias=bias)
            nn.init.normal_(self.reduce.weight, std=math.sqrt(2.0 / max(1, self.in_chans)))
            if bias:
                nn.init.zeros_(self.reduce.bias)
        else:
            self.reduce = nn.Identity()

        # The upsample, positional-embedding query and cross-resolution attention
        # only exist when the decoder actually changes resolution. When
        # inp_shape == out_shape the decoder degenerates to the pointwise
        # projection (mirrors the encoder).
        if self._needs_resample:
            # Bilinear-spherical (or SHT) upsample latent -> output grid: both the
            # residual base and the source of the attention query.
            if upsample_sht:
                sht = th.RealSHT(*inp_shape, grid=grid_in).float()
                isht = th.InverseRealSHT(
                    *out_shape,
                    lmax=sht.lmax,
                    mmax=sht.mmax,
                    grid=grid_out,
                ).float()
                self.upsample = nn.Sequential(sht, isht)
            else:
                self.upsample = th.ResampleS2(
                    *inp_shape, *out_shape, grid_in=grid_in, grid_out=grid_out
                )

            # Position-aware query on the OUTPUT grid (non-persistent buffer, not
            # in the state dict; recomputed for whatever grid is requested).
            self.pos_embed: Optional[nn.Module] = _SpectralPositionEmbedding(
                out_shape, self.attn_dim, grid_out, max_degree=pos_embed_max_degree,
            )

            # Cross-resolution UPSAMPLING attention: K/V at the latent (inp_shape),
            # Q at the output (out_shape). Requires nlon_out % nlon_in == 0.
            theta_cutoff = _neighborhood_radius_rad(out_shape[0], theta_cutoff_factor)
            self.attn: Optional[nn.Module] = th.NeighborhoodAttentionS2(
                in_channels=self.attn_dim,
                in_shape=inp_shape,
                out_shape=out_shape,
                grid_in=grid_in,
                grid_out=grid_out,
                num_heads=num_heads,
                theta_cutoff=theta_cutoff,
                use_qknorm=qk_norm,
                bias=bias,
                out_channels=self.attn_dim,
                optimized_kernel=attn_optimized_kernel,
            )
        else:
            self.upsample = nn.Identity()
            self.pos_embed = None
            self.attn = None

        self.project = nn.Conv2d(self.attn_dim, out_chans, kernel_size=1, bias=bias)
        nn.init.normal_(self.project.weight, std=math.sqrt(2.0 / max(1, self.attn_dim)))
        if bias:
            nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode ``x`` of shape :math:`(B, C_{in}, h, w)` to :math:`(B, C_{out}, H_{out}, W_{out})`."""
        x = self.reduce(x)  # (B, attn_dim, h, w)
        if not self._needs_resample:
            # No resolution change: pointwise channel projection only.
            return self.project(x)
        dtype = x.dtype
        with amp.autocast(device_type=x.device.type, enabled=False):
            kv = x.float()                          # (B, attn_dim, h, w)
            base = self.upsample(kv)                # (B, attn_dim, H, W)
            q = self.pos_embed(base)                # base + PE (position-aware query)
            delta = self.attn(q, key=kv, value=kv)  # (B, attn_dim, H, W)
            x = (base + delta).to(dtype)            # residual on the bilinear base
        return self.project(x)                      # (B, out_chans, H, W)


# -----------------------------------------------------------------------------
# Positional embeddings
# -----------------------------------------------------------------------------


class _SpectralPositionEmbedding(nn.Module):
    r"""Spherical-harmonic positional embedding (channel-wise basis function).

    Each channel is one real spherical harmonic on the ``(h, w)`` grid,
    normalised to unit max amplitude. Cheap, deterministic, and well-suited to
    spherical transformer pretraining.

    Degree coverage (``max_degree``):

    - ``None`` (default): channel ``i`` is the ``i``-th harmonic in the standard
      ``(l, m)`` enumeration, so for ``num_chans`` channels the degree only
      reaches ``l ~ floor(sqrt(num_chans))`` (low-frequency only).
    - integer: spread ``num_chans`` harmonics evenly (in enumeration index) over
      degrees ``0 .. max_degree`` (clamped to the grid bandlimit), so the
      embedding carries high-frequency components able to resolve fine position
      differences while still including the smooth low-degree modes.

    Parameters
    ----------
    grid_shape : tuple of int
        Grid shape :math:`(h, w)` the embedding is defined on.
    num_chans : int
        Number of channels (one harmonic per channel).
    grid : str
        Quadrature grid type.
    max_degree : int, optional, default=None
        Maximum spherical-harmonic degree to spread the channels over.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        num_chans: int,
        grid: str,
        max_degree: Optional[int] = None,
    ):
        super().__init__()
        H, W = int(grid_shape[0]), int(grid_shape[1])
        isht = th.InverseRealSHT(nlat=H, nlon=W, grid=grid)

        # Build the list of (l, m) harmonics assigned to each channel.
        if max_degree is None:
            # The first ``num_chans`` harmonics (all low-degree).
            pairs = []
            for i in range(num_chans):
                l = math.floor(math.sqrt(i))
                m = i - l * l - l
                pairs.append((l, m))
        else:
            # Spread across degrees 0..L (clamped to the grid bandlimit), so
            # high-frequency harmonics are included for sharp localisation.
            L = max(0, min(int(max_degree), int(isht.lmax) - 1))
            all_pairs = [
                (l, m)
                for l in range(L + 1)
                for m in range(-l, l + 1)
                if abs(m) < int(isht.mmax)
            ]
            if len(all_pairs) >= num_chans:
                idx = (
                    torch.linspace(0, len(all_pairs) - 1, num_chans)
                    .round()
                    .long()
                    .tolist()
                )
                pairs = [all_pairs[j] for j in idx]
            else:
                # Fewer harmonics than channels (tiny grids): cycle through.
                pairs = [all_pairs[j % len(all_pairs)] for j in range(num_chans)]

        with torch.no_grad():
            pos_freq = torch.zeros(
                1, num_chans, isht.lmax, isht.mmax, dtype=torch.complex64
            )
            for i, (l, m) in enumerate(pairs):
                # Guard against out-of-range indices on small grids.
                if l >= isht.lmax or abs(m) >= isht.mmax:
                    continue
                if m < 0:
                    pos_freq[0, i, l, -m] = 1.0j
                else:
                    pos_freq[0, i, l, m] = 1.0
            pos_embed = isht(pos_freq)
            pos_embed = pos_embed / (
                pos_embed.abs().amax(dim=(-1, -2), keepdim=True) + 1.0e-8
            )
        self.register_buffer("position_embeddings", pos_embed.float(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the (broadcast) positional embedding to ``x``."""
        return x + self.position_embeddings


def _build_pos_embedding(
    kind: str,
    grid_shape: Tuple[int, int],
    num_chans: int,
    grid: str,
    max_degree: Optional[int] = None,
) -> nn.Module:
    """Build the positional-embedding module selected by ``kind``.

    Parameters
    ----------
    kind : str
        One of ``"none"`` or ``"spectral"``.
    grid_shape : tuple of int
        Grid shape :math:`(h, w)`.
    num_chans : int
        Number of channels.
    grid : str
        Quadrature grid type (used by the spectral embedding).
    max_degree : int, optional, default=None
        Maximum spherical-harmonic degree for the spectral embedding.

    Returns
    -------
    nn.Module
        The positional-embedding module (or ``nn.Identity`` for ``"none"``).
    """
    kind = (kind or "none").lower()
    if kind == "none":
        return nn.Identity()
    if kind == "spectral":
        return _SpectralPositionEmbedding(
            grid_shape, num_chans, grid, max_degree=max_degree
        )
    raise ValueError(
        f"Unknown pos_embed='{kind}'. Supported: 'none', 'spectral'."
    )


# -----------------------------------------------------------------------------
# Spherical transformer block
# -----------------------------------------------------------------------------


def _make_norm(kind: str, num_chans: int) -> nn.Module:
    """Build the normalization module selected by ``kind``.

    Parameters
    ----------
    kind : str
        One of ``"none"``, ``"layer_norm"``, ``"instance_norm"``.
    num_chans : int
        Number of channels to normalize over.

    Returns
    -------
    nn.Module
        The normalization module (or ``nn.Identity`` for ``"none"``).
    """
    kind = (kind or "none").lower()
    if kind == "none":
        return nn.Identity()
    if kind == "layer_norm":
        # LayerNorm over channels at every spatial location, expressed as a
        # channel-first-friendly GroupNorm with groups=1 (equivalent).
        return nn.GroupNorm(num_groups=1, num_channels=num_chans, eps=1e-6, affine=True)
    if kind == "instance_norm":
        return nn.InstanceNorm2d(
            num_features=num_chans, eps=1e-6, affine=True, track_running_stats=False
        )
    raise ValueError(
        f"Unknown normalization_layer='{kind}'. Supported: 'none', 'layer_norm', 'instance_norm'."
    )


class SphericalTransformerBlock(nn.Module):
    r"""Pre-norm spherical transformer block.

    The spatial mixer (attention) keeps the channel count constant at
    ``in_chans``, and the MLP is the component that changes channels
    (``in_chans -> out_chans``). When ``in_chans != out_chans`` the MLP residual
    is disabled (there is no channel-matching identity path).

    Order of operations:

    .. code-block:: text

        residual = x
        x = norm0(x)
        x = self_attn(x)               # in_chans -> in_chans
        x = residual + drop_path(x)
        residual = x
        x = norm1(x)
        x = mlp(x)                     # in_chans -> out_chans
        x = residual + drop_path(x)    # only if in_chans == out_chans

    Parameters
    ----------
    in_shape : tuple of int
        Grid shape :math:`(h, w)` the block operates on.
    grid : str
        Quadrature grid type.
    in_chans : int
        Number of input channels (must be divisible by ``num_heads``).
    out_chans : int
        Number of output channels.
    num_heads : int
        Number of attention heads.
    attention_mode : str, optional, default="neighborhood"
        Either ``"neighborhood"`` (local geodesic-cap attention) or ``"global"``
        (full quadrature-weighted attention).
    attn_theta_cutoff : float, optional, default=None
        Neighborhood radius in radians (only used for ``"neighborhood"``).
    mlp_ratio : float, optional, default=2.0
        Ratio of the MLP hidden dimension to ``in_chans``.
    mlp_drop_rate : float, optional, default=0.0
        Dropout rate inside the MLP.
    attn_drop_rate : float, optional, default=0.0
        Dropout rate inside the global-attention softmax (no-op for neighborhood).
    path_drop_rate : float, optional, default=0.0
        Stochastic-depth (drop-path) rate.
    act_layer : nn.Module, optional, default=nn.GELU
        Activation layer constructor.
    normalization_layer : str, optional, default="layer_norm"
        One of ``"none"``, ``"layer_norm"``, ``"instance_norm"``.
    bias : bool, optional, default=True
        Whether the linear projections use a bias.
    use_mlp : bool, optional, default=True
        Whether to include the MLP sublayer. If ``False``, ``in_chans`` must
        equal ``out_chans``.
    qk_norm : bool, optional, default=False
        Whether to apply RMSNorm to the query/key projections.
    attn_optimized_kernel : bool, optional, default=True
        Whether to use the optimized CUDA neighborhood-attention kernel.
    checkpointing_level : int, optional, default=0
        Activation-checkpointing aggressiveness (``>= 2`` checkpoints the MLP).
    """

    def __init__(
        self,
        in_shape: Tuple[int, int],
        grid: str,
        in_chans: int,
        out_chans: int,
        num_heads: int,
        attention_mode: str = "neighborhood",
        attn_theta_cutoff: Optional[float] = None,
        mlp_ratio: float = 2.0,
        mlp_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        path_drop_rate: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        normalization_layer: str = "layer_norm",
        bias: bool = True,
        use_mlp: bool = True,
        qk_norm: bool = False,
        attn_optimized_kernel: bool = True,
        checkpointing_level: int = 0,
        film_dim: int = 0,
    ):
        super().__init__()

        if in_chans % num_heads != 0:
            raise ValueError(
                f"SphericalTransformerBlock: in_chans ({in_chans}) must be "
                f"divisible by num_heads ({num_heads})."
            )

        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.attention_mode = attention_mode
        self.checkpointing_level = int(checkpointing_level)

        self.norm0 = _make_norm(normalization_layer, in_chans)
        if attention_mode == "neighborhood":
            self.self_attn = th.NeighborhoodAttentionS2(
                in_channels=in_chans,
                in_shape=in_shape,
                out_shape=in_shape,
                grid_in=grid,
                grid_out=grid,
                num_heads=num_heads,
                theta_cutoff=attn_theta_cutoff,
                use_qknorm=qk_norm,
                bias=bias,
                out_channels=in_chans,
                optimized_kernel=attn_optimized_kernel,
            )
        elif attention_mode == "global":
            self.self_attn = th.AttentionS2(
                in_channels=in_chans,
                num_heads=num_heads,
                in_shape=in_shape,
                out_shape=in_shape,
                grid_in=grid,
                grid_out=grid,
                use_qknorm=qk_norm,
                bias=bias,
                out_channels=in_chans,
                drop_rate=attn_drop_rate,
            )
        else:
            raise ValueError(
                f"Unknown attention_mode='{attention_mode}'. Supported: 'neighborhood', 'global'."
            )
        self.drop_path0 = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

        self.norm1 = _make_norm(normalization_layer, in_chans)
        if use_mlp:
            self.mlp = MLP(
                in_features=in_chans,
                out_features=out_chans,
                hidden_features=int(in_chans * mlp_ratio),
                act_layer=act_layer,
                drop_rate=mlp_drop_rate,
                drop_type="features",
                checkpointing=(self.checkpointing_level >= 2),
            )
        else:
            self.mlp = None
            if in_chans != out_chans:
                raise ValueError(
                    f"SphericalTransformerBlock: use_mlp=False requires "
                    f"in_chans ({in_chans}) == out_chans ({out_chans}); "
                    f"with no MLP there is no channel-changing path."
                )
        self.drop_path1 = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

        # A channel-changing MLP cannot share the residual stream, so the MLP
        # residual is only applied when in_chans == out_chans.
        self._mlp_residual = self.in_chans == self.out_chans

        # Optional conditional (FiLM) modulation of the two pre-norm streams. Each
        # head maps a shared per-sample conditioning embedding -> per-channel
        # (scale, shift). The heads are ZERO-initialised so that at init (and
        # whenever the conditioning embedding is absent) the modulation is the
        # identity ``h -> h * (1 + 0) + 0 = h``. This makes warm-starting from a
        # deterministic checkpoint EXACT: the conditional path contributes nothing
        # until trained. (FiLM-Ensemble, NeurIPS 2022; AIFS-CRPS conditioning.)
        self.film_dim = int(film_dim)
        if self.film_dim > 0:
            self.film0 = nn.Linear(self.film_dim, 2 * in_chans)
            self.film1 = nn.Linear(self.film_dim, 2 * in_chans)
            for _h in (self.film0, self.film1):
                nn.init.zeros_(_h.weight)
                nn.init.zeros_(_h.bias)
        else:
            self.film0 = None
            self.film1 = None

    def _attn_forward(self, x: torch.Tensor) -> torch.Tensor:
        # NeighborhoodAttentionS2's optimized CUDA kernel only supports FP32,
        # so we disable autocast around it. AttentionS2 handles autocast natively.
        if self.attention_mode == "neighborhood":
            dtype = x.dtype
            with amp.autocast(device_type=x.device.type, enabled=False):
                return self.self_attn(x.float()).to(dtype)
        return self.self_attn(x)

    @staticmethod
    def _apply_film(h: torch.Tensor, head: nn.Module, film_embed: torch.Tensor) -> torch.Tensor:
        gamma, beta = head(film_embed).chunk(2, dim=-1)          # (B, C) each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).to(h.dtype)    # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).to(h.dtype)
        return h * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, film_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the pre-norm attention + MLP block to ``x`` of shape :math:`(B, C, h, w)`.

        If ``film_embed`` is provided and FiLM heads exist, each pre-norm stream is
        modulated by a per-channel (scale, shift) derived from ``film_embed``.
        """
        residual = x
        h = self.norm0(x)
        if self.film0 is not None and film_embed is not None:
            h = self._apply_film(h, self.film0, film_embed)
        h = self._attn_forward(h)
        x = residual + self.drop_path0(h)

        if self.mlp is not None:
            residual = x
            h = self.norm1(x)
            if self.film1 is not None and film_embed is not None:
                h = self._apply_film(h, self.film1, film_embed)
            h = self.mlp(h)
            h = self.drop_path1(h)
            x = residual + h if self._mlp_residual else h
        return x


# -----------------------------------------------------------------------------
# PrecipAttentionNet
# -----------------------------------------------------------------------------


@dataclass
class MetaData(ModelMetaData):
    """Metadata for :class:`PrecipAttentionNet`."""

    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = True
    onnx_cpu: bool = False
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class PrecipAttentionNet(Module):
    r"""Fully attention-based spherical transformer (S2Transformer) backbone.

    An encoder / processor / decoder network for gridded geophysical fields
    (developed for precipitation diagnostics). The internal (processor) grid is
    either derived from ``scale_factor`` (``h, w = inp_shape // scale_factor``)
    or set explicitly via ``latent_shape=(h, w)``, which takes precedence. The
    latent longitude ``w`` MUST divide the input longitude (a hard constraint of
    the spherical-attention p-shift); latitude ``h`` is free.

    ``use_encoder=False`` / ``use_decoder=False`` skip the spherical-attention
    encoder / decoder (no spatial up/downsampling), so the processor grid must
    equal the input grid (``scale_factor=1`` or ``latent_shape=inp_shape``). In
    that case the first block's input channel count is ``in_channels`` (instead
    of ``embed_dim``) and/or the last block's output channel count is
    ``out_channels``; the channel change happens inside the block's MLP, and the
    MLP residual is disabled when ``in_chans != out_chans``. Because spherical
    attention requires ``num_heads | in_channels``, ``use_encoder=False``
    constrains ``num_heads`` to divide ``in_channels``.

    Parameters
    ----------
    model_grid_type : str, optional, default="equiangular"
        Quadrature grid type of the input/output grids.
    sht_grid_type : str, optional, default="legendre-gauss"
        Quadrature grid type of the internal (processor) grid.
    inp_shape : tuple of int, optional, default=(721, 1440)
        Input grid shape :math:`(H_{in}, W_{in})`.
    out_shape : tuple of int, optional, default=(721, 1440)
        Output grid shape :math:`(H_{out}, W_{out})`.
    in_channels : int, optional, default=27
        Number of input channels.
    out_channels : int, optional, default=1
        Number of output channels.
    scale_factor : int, optional, default=8
        Spatial downsampling factor for the processor grid (ignored when
        ``latent_shape`` is given).
    latent_shape : tuple of int, optional, default=None
        Explicit processor grid shape :math:`(h, w)` (takes precedence over
        ``scale_factor``).
    upsample_sht : bool, optional, default=False
        Whether the decoder upsamples with an SHT round-trip.
    channel_names : list of str, optional, default=None
        Optional channel names (metadata only).
    n_history : int, optional, default=0
        Number of history steps. Only ``0`` is currently supported.
    embed_dim : int, optional, default=128
        Processor (latent) channel width. Must be divisible by ``num_heads``.
    num_layers : int, optional, default=4
        Number of processor :class:`SphericalTransformerBlock` layers.
    num_heads : int, optional, default=4
        Number of attention heads.
    attention_mode : str or list of str, optional, default="neighborhood"
        ``"neighborhood"``, ``"global"``, or a per-layer list of length
        ``num_layers``.
    attn_theta_cutoff_factor : float, optional, default=2.0
        Processor neighborhood radius in units of latitude rings.
    attn_drop_rate : float, optional, default=0.0
        Dropout applied inside the global-attention softmax.
    attn_optimized_kernel : bool, optional, default=True
        Whether to use the optimized CUDA neighborhood-attention kernel.
    qk_norm : bool, optional, default=False
        Whether to apply RMSNorm to the query/key projections.
    mlp_ratio : float, optional, default=2.0
        Ratio of the MLP hidden dimension to the block input width.
    activation_function : str, optional, default="gelu"
        One of ``"relu"``, ``"gelu"``, ``"silu"``.
    pos_drop_rate : float, optional, default=0.0
        Dropout applied to the positional embedding.
    path_drop_rate : float, optional, default=0.0
        Maximum stochastic-depth rate (linearly ramped across layers).
    mlp_drop_rate : float, optional, default=0.0
        Dropout rate inside the MLP sublayers.
    pos_embed : str, optional, default="spectral"
        One of ``"none"`` or ``"spectral"`` (a fixed, non-learnable
        spherical-harmonic embedding).
    pos_embed_max_degree : int, optional, default=None
        Maximum spherical-harmonic degree for the ``"spectral"`` embedding
        (higher = finer positional resolution). ``None`` uses the low-degree
        default enumeration.
    normalization_layer : str, optional, default="layer_norm"
        One of ``"none"``, ``"layer_norm"``, ``"instance_norm"``.
    encoder_theta_cutoff_factor : float, optional, default=1.0
        Encoder neighborhood radius in units of latitude rings (input grid).
    decoder_theta_cutoff_factor : float, optional, default=1.0
        Decoder neighborhood radius in units of latitude rings (output grid).
    use_encoder : bool, optional, default=True
        Whether to include the spherical-attention encoder.
    use_decoder : bool, optional, default=True
        Whether to include the spherical-attention decoder.
    big_skip : bool, optional, default=False
        Whether to add a learned 1x1 residual from the input to the output.
    bias : bool, optional, default=True
        Whether the linear projections use a bias.
    use_mlp : bool, optional, default=True
        Whether the processor blocks include the MLP sublayer.
    checkpointing_level : int, optional, default=0
        Activation-checkpointing aggressiveness.
    noise_mode : str, optional, default=None
        Stochastic conditioning mode. ``None`` disables it (purely deterministic).
        ``"film"`` enables FiLM conditioning: a per-sample latent modulates every
        processor block's pre-norm streams via zero-init (scale, shift) heads, so
        a deterministic checkpoint warm-starts exactly and gains diversity only
        once trained. Pass the latent to ``forward(..., film_latent=z)``.
    noise_channels : int, optional, default=1
        Reserved for API parity.
    film_latent_dim : int, optional, default=64
        Dimension of the per-sample FiLM latent (used only when ``noise_mode='film'``).
    film_hidden : int, optional, default=0
        Hidden/embedding width of the FiLM conditioning MLP. ``0`` uses
        ``max(64, embed_dim)``.

    Example
    -------
    >>> import torch
    >>> from physicsnemo.experimental.models.s2transformer import PrecipAttentionNet
    >>> model = PrecipAttentionNet(  # doctest: +SKIP
    ...     inp_shape=(32, 64),
    ...     out_shape=(32, 64),
    ...     in_channels=4,
    ...     out_channels=1,
    ...     latent_shape=(16, 32),
    ...     embed_dim=16,
    ...     num_layers=2,
    ...     num_heads=4,
    ... )
    >>> model(torch.randn(1, 4, 32, 64)).shape  # doctest: +SKIP
    torch.Size([1, 1, 32, 64])
    """

    def __init__(
        self,
        model_grid_type: str = "equiangular",
        sht_grid_type: str = "legendre-gauss",
        inp_shape: Tuple[int, int] = (721, 1440),
        out_shape: Tuple[int, int] = (721, 1440),
        in_channels: int = 27,
        out_channels: int = 1,
        scale_factor: int = 8,
        latent_shape: Optional[Tuple[int, int]] = None,
        upsample_sht: bool = False,
        channel_names: Optional[List[str]] = None,
        n_history: int = 0,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        attention_mode="neighborhood",
        attn_theta_cutoff_factor: float = 2.0,
        attn_drop_rate: float = 0.0,
        attn_optimized_kernel: bool = True,
        qk_norm: bool = False,
        mlp_ratio: float = 2.0,
        activation_function: str = "gelu",
        pos_drop_rate: float = 0.0,
        path_drop_rate: float = 0.0,
        mlp_drop_rate: float = 0.0,
        pos_embed: str = "spectral",
        pos_embed_max_degree: Optional[int] = None,
        normalization_layer: str = "layer_norm",
        encoder_theta_cutoff_factor: float = 1.0,
        decoder_theta_cutoff_factor: float = 1.0,
        use_encoder: bool = True,
        use_decoder: bool = True,
        big_skip: bool = False,
        bias: bool = True,
        use_mlp: bool = True,
        checkpointing_level: int = 0,
        noise_mode: Optional[str] = None,
        noise_channels: int = 1,
        film_latent_dim: int = 64,
        film_hidden: int = 0,
        **kwargs,
    ):
        super().__init__(meta=MetaData())

        if not TORCH_HARMONICS_AVAILABLE:
            raise ImportError(
                "PrecipAttentionNet requires the optional dependency "
                "'torch_harmonics' (>=0.7.0), which is not installed.\n"
                "Install with:\n  pip install torch_harmonics>=0.7.0\n"
                "  pip install nvidia-physicsnemo[harmonics]"
            )
        if n_history != 0:
            raise ValueError("PrecipAttentionNet currently only supports n_history=0.")
        if noise_mode not in (None, "film"):
            raise NotImplementedError(
                f"PrecipAttentionNet supports noise_mode in (None, 'film'); "
                f"got {noise_mode!r}."
            )

        self.inp_shape = tuple(inp_shape)
        self.out_shape = tuple(out_shape)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.use_encoder = bool(use_encoder)
        self.use_decoder = bool(use_decoder)
        self.big_skip = bool(big_skip)
        self.checkpointing_level = int(checkpointing_level)

        # FiLM ("film") stochastic conditioning: a per-sample latent vector is
        # turned into a shared embedding that every processor block's zero-init
        # FiLM head maps to per-channel (scale, shift). It does NOT change the
        # input channel count, so a deterministic (noise_mode=None) checkpoint
        # warm-starts exactly. noise_mode=None disables it entirely.
        self.noise_mode = noise_mode
        self.film_latent_dim = int(film_latent_dim) if noise_mode == "film" else 0
        self._film_dim = 0
        if self.noise_mode == "film":
            self._film_dim = int(film_hidden) if film_hidden else max(64, self.embed_dim)
            self.film_embed = nn.Sequential(
                nn.Linear(self.film_latent_dim, self._film_dim),
                nn.SiLU(),
                nn.Linear(self._film_dim, self._film_dim),
                nn.LayerNorm(self._film_dim),
            )
        else:
            self.film_embed = None

        if activation_function == "relu":
            act_layer = nn.ReLU
        elif activation_function == "gelu":
            act_layer = nn.GELU
        elif activation_function == "silu":
            act_layer = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # Internal (processor) spatial shape. Two ways to specify it:
        #   1. latent_shape=(h, w) - explicit, takes precedence.
        #   2. scale_factor - h,w = inp_shape // scale_factor.
        # The latent longitude (w) MUST divide the input longitude (a hard
        # constraint of the spherical-attention p-shift kernel); latitude (h)
        # is free.
        if latent_shape is not None:
            self.h = int(latent_shape[0])
            self.w = int(latent_shape[1])
        else:
            self.h = int(self.inp_shape[0] // scale_factor)
            self.w = int(self.inp_shape[1] // scale_factor)
        if self.use_encoder and self.inp_shape[1] % self.w != 0:
            raise ValueError(
                f"PrecipAttentionNet: latent longitude w={self.w} must divide "
                f"input longitude W={self.inp_shape[1]} (spherical-attention "
                f"p-shift). Pick a w that divides {self.inp_shape[1]}."
            )

        # Encoder/decoder: when off they are pure nn.Identity() and the channel
        # routing in_channels -> embed_dim -> out_channels is pushed into the
        # first/last processor block (the block's MLP changes channels). The
        # spatial shape cannot change without the attention encoder/decoder, so
        # use_encoder=False / use_decoder=False require the processor grid to
        # equal the input grid.
        if self.use_encoder:
            self.encoder = AttentionEncoder(
                inp_shape=self.inp_shape,
                out_shape=(self.h, self.w),
                in_chans=self.in_channels,
                out_chans=self.embed_dim,
                grid_in=model_grid_type,
                grid_out=sht_grid_type,
                num_heads=int(num_heads),
                theta_cutoff_factor=encoder_theta_cutoff_factor,
                qk_norm=qk_norm,
                bias=bias,
                attn_optimized_kernel=attn_optimized_kernel,
            )
        else:
            if (self.h, self.w) != tuple(self.inp_shape):
                raise ValueError(
                    "use_encoder=False with PrecipAttentionNet requires the "
                    "processor grid to match the input grid (set scale_factor=1). "
                    f"Got inp_shape={self.inp_shape} but processor grid is "
                    f"({self.h}, {self.w})."
                )
            self.encoder = nn.Identity()

        if self.use_decoder:
            self.decoder = AttentionDecoder(
                inp_shape=(self.h, self.w),
                out_shape=self.out_shape,
                in_chans=self.embed_dim,
                out_chans=self.out_channels,
                grid_in=sht_grid_type,
                grid_out=model_grid_type,
                num_heads=int(num_heads),
                theta_cutoff_factor=decoder_theta_cutoff_factor,
                qk_norm=qk_norm,
                bias=bias,
                attn_optimized_kernel=attn_optimized_kernel,
                upsample_sht=upsample_sht,
            )
        else:
            if (self.h, self.w) != tuple(self.out_shape):
                raise ValueError(
                    "use_decoder=False with PrecipAttentionNet requires the "
                    "processor grid to match the output grid (set scale_factor=1). "
                    f"Got out_shape={self.out_shape} but processor grid is "
                    f"({self.h}, {self.w})."
                )
            self.decoder = nn.Identity()

        # Resolve per-block attention modes (allow list-of-strings for mix-and-match).
        if isinstance(attention_mode, str):
            modes = [attention_mode] * self.num_layers
        else:
            modes = list(attention_mode)
            if len(modes) != self.num_layers:
                raise ValueError(
                    f"attention_mode list length ({len(modes)}) must equal "
                    f"num_layers ({self.num_layers})."
                )

        attn_theta_cutoff = _neighborhood_radius_rad(self.h, attn_theta_cutoff_factor)

        # Per-block channel routing:
        #   first block:  in_ch = in_channels   if use_encoder=False else embed_dim
        #   last block:   out_ch = out_channels if use_decoder=False else embed_dim
        #   middle blocks always operate at embed_dim. The channel change happens
        #   inside SphericalTransformerBlock's MLP.
        def _block_in_ch(i: int) -> int:
            if i == 0 and not self.use_encoder:
                return self.in_channels
            return self.embed_dim

        def _block_out_ch(i: int) -> int:
            if i == self.num_layers - 1 and not self.use_decoder:
                return self.out_channels
            return self.embed_dim

        # The positional embedding lives just before the first processor block,
        # so it has to match that block's input channel count.
        first_in_ch = _block_in_ch(0)
        self.pos_drop = nn.Dropout(p=pos_drop_rate) if pos_drop_rate > 0.0 else nn.Identity()
        self.pos_embed = _build_pos_embedding(
            pos_embed,
            (self.h, self.w),
            first_in_ch,
            sht_grid_type,
            max_degree=pos_embed_max_degree,
        )

        dpr = [
            float(x) for x in torch.linspace(0, path_drop_rate, max(1, self.num_layers))
        ]
        self.blocks = nn.ModuleList(
            [
                SphericalTransformerBlock(
                    in_shape=(self.h, self.w),
                    grid=sht_grid_type,
                    in_chans=_block_in_ch(i),
                    out_chans=_block_out_ch(i),
                    num_heads=int(num_heads),
                    attention_mode=modes[i],
                    attn_theta_cutoff=attn_theta_cutoff,
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    path_drop_rate=dpr[i],
                    act_layer=act_layer,
                    normalization_layer=normalization_layer,
                    bias=bias,
                    use_mlp=use_mlp,
                    qk_norm=qk_norm,
                    attn_optimized_kernel=attn_optimized_kernel,
                    checkpointing_level=self.checkpointing_level,
                    film_dim=self._film_dim,
                )
                for i in range(self.num_layers)
            ]
        )

        if self.big_skip:
            self.residual_transform = nn.Conv2d(
                self.in_channels, self.out_channels, 1, bias=False
            )
            scale = math.sqrt(0.5 / max(1, self.in_channels))
            nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input to the processor grid."""
        return self.encoder(x.contiguous())

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the processor output back to the output grid."""
        return self.decoder(x)

    def processor_blocks(
        self, x: torch.Tensor, film_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply positional embedding and the stack of processor blocks."""
        x = self.pos_embed(self.pos_drop(x))
        for blk in self.blocks:
            if self.checkpointing_level >= 3:
                x = checkpoint(blk, x, film_embed, use_reentrant=False)
            else:
                x = blk(x, film_embed=film_embed)
        return x

    def forward(
        self,
        x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        film_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the full encoder / processor / decoder network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, C_{in}, H_{in}, W_{in})`.
        noise : torch.Tensor, optional, default=None
            Reserved for API parity; must be ``None``.
        film_latent : torch.Tensor, optional, default=None
            Per-sample conditioning latent of shape :math:`(B, film\\_latent\\_dim)`,
            used only when ``noise_mode='film'``. When absent, the zero-init FiLM
            heads make the network reproduce its deterministic mean map.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, C_{out}, H_{out}, W_{out})`.
        """
        if noise is not None:
            raise NotImplementedError(
                "PrecipAttentionNet does not (yet) support noise conditioning."
            )
        film_embed = None
        if self.noise_mode == "film" and film_latent is not None:
            film_embed = self.film_embed(
                film_latent.to(dtype=self.film_embed[0].weight.dtype)
            )
        residual = x.contiguous() if self.big_skip else None

        if self.checkpointing_level >= 1:
            x = checkpoint(self.encode, x, use_reentrant=False)
        else:
            x = self.encode(x)

        x = self.processor_blocks(x, film_embed=film_embed)

        if self.checkpointing_level >= 1:
            x = checkpoint(self.decode, x, use_reentrant=False)
        else:
            x = self.decode(x)

        if self.big_skip:
            x = x + self.residual_transform(residual)
        return x
