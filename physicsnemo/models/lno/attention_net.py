# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fully attention-based variant of :class:`PrecipNeuralOperatorNet`.

The topology mirrors the existing precip backbone so this module is a drop-in
replacement inside the ``precip_diagnostic`` pipeline (same constructor kwargs
for ``in_channels``/``out_channels``/``inp_shape``/``model_grid_type`` etc.),
but every internal mixer is a spherical transformer block built around
torch-harmonics' :class:`AttentionS2` / :class:`NeighborhoodAttentionS2`
(Bonev et al., NeurIPS 2025, https://arxiv.org/abs/2505.11157):

.. code-block:: text

    DiscreteContinuousEncoder (in_channels -> embed_dim,  H_in,W_in -> h,w)
        + optional positional embedding on the (h, w) internal grid
        -> [SphericalTransformerBlock] x num_layers
    DiscreteContinuousDecoder (embed_dim -> out_channels, h,w -> H_in,W_in)
        + optional big-skip residual from the input

Each ``SphericalTransformerBlock`` is the canonical pre-norm transformer:
``x = x + drop_path(self_attn(norm0(x)))`` followed by
``x = x + drop_path(mlp(norm1(x)))``. The attention is quadrature-aware on the
sphere (softmax weights include the geodesic Jacobian), so the layer remains
SO(3)-equivariant in expectation and is well-behaved at the poles.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
import torch_harmonics as th
from torch.utils.checkpoint import checkpoint
from torch_harmonics import InverseRealSHT

from physicsnemo.core import ModelMetaData, Module

from ._layers import DropPath, MLP


def _neighborhood_radius_rad(nlat: int, factor: float) -> float:
    """Geodesic-cap radius (radians) for neighborhood attention.

    Expressed in units of latitude spacing on an equiangular-style grid:
    ``radius = factor * pi / (nlat - 1)``. ``factor = 1.0`` therefore covers
    one latitude ring, ``factor = 4.0`` four rings, etc. This replaces the
    DISCO-convolution heuristic (which mixed a ``kernel_shape + 1`` term with
    a filter-basis-dependent scale) — attention has no kernel basis, so the
    radius is just a geodesic angle.
    """
    return float(factor) * math.pi / float(max(int(nlat) - 1, 1))


# -----------------------------------------------------------------------------
# Attention encoder / decoder
# -----------------------------------------------------------------------------


class AttentionEncoder(nn.Module):
    """Spherical-attention encoder: ``(in_chans, H_in, W_in) -> (out_chans, h, w)``.

    Pipeline:

    1. Pointwise 1x1 lift ``in_chans -> out_chans`` on the input grid.
    2. :class:`NeighborhoodAttentionS2` **cross-resolution downsampling**:
       the attention itself aggregates from the full-resolution K/V tokens to
       the coarse Q positions, so no explicit resampling module is needed.

       Following the updated torch-harmonics API, Q must be at ``out_shape``
       and K/V at ``in_shape``.  Q is initialised by
       ``F.adaptive_avg_pool2d`` — a cheap positional pointer that defines
       *where* each output token is; all spatial information is gathered
       from the full-resolution K/V by the attention.

       An optional positional embedding (``pos_embed``, assigned externally)
       is added to Q after the pool and before the attention so the output
       tokens know their position on the sphere.

    Requires ``nlon_in % nlon_out == 0`` (a hard constraint of the
    underlying spherical attention kernel).
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
    ):
        super().__init__()
        if out_chans % num_heads != 0:
            raise ValueError(
                f"AttentionEncoder: out_chans ({out_chans}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.lift = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=bias)
        nn.init.normal_(self.lift.weight, std=math.sqrt(2.0 / max(1, in_chans)))
        if bias:
            nn.init.zeros_(self.lift.bias)

        # Spherically-correct bilinear downsample to the latent grid.
        # Used both as the residual base and as Q seed for the cross-attention.
        self._needs_resample = (inp_shape != out_shape)
        if self._needs_resample:
            self.resample = th.ResampleS2(
                inp_shape[0], inp_shape[1],
                out_shape[0], out_shape[1],
                grid_in=grid_in,
                grid_out=grid_out,
            )

        # Optional spatial PE applied to Q only (assigned externally).
        self.pos_embed: Optional[nn.Module] = None

        # Cross-resolution attention: K/V at inp_shape, Q at out_shape.
        # Output is added as a residual on top of the bilinear downsample.
        theta_cutoff = _neighborhood_radius_rad(inp_shape[0], theta_cutoff_factor)
        self.attn = th.NeighborhoodAttentionS2(
            in_channels=out_chans,
            in_shape=inp_shape,
            out_shape=out_shape,
            grid_in=grid_in,
            grid_out=grid_out,
            num_heads=num_heads,
            theta_cutoff=theta_cutoff,
            use_qknorm=qk_norm,
            bias=bias,
            out_channels=out_chans,
            optimized_kernel=attn_optimized_kernel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)    # (B, D, H, W)
        dtype = x.dtype
        with amp.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            if self._needs_resample:
                # Residual base: spherically-correct bilinear downsample.
                q_base = self.resample(x_f)                      # (B, D, h, w)
                # Q for attention: add spatial PE so attention is position-aware.
                q_attn = self.pos_embed(q_base) if self.pos_embed is not None else q_base
                # Cross-attention correction: what the full-resolution input adds
                # beyond the bilinear base.
                delta = self.attn(q_attn, key=x_f, value=x_f)   # (B, D, h, w)
                x = (q_base + delta).to(dtype)                   # residual
            else:
                # Same shape: plain self-attention, no downsampling.
                q = self.pos_embed(x_f) if self.pos_embed is not None else x_f
                x = (x_f + self.attn(q)).to(dtype)
        return x


class AttentionDecoder(nn.Module):
    """Spherical-attention decoder: ``(in_chans, h, w) -> (out_chans, H_out, W_out)``.

    Pipeline:

    1. Upsample with :class:`ResampleS2` (SLERP-style on the sphere) or with
       an :class:`SHT` round-trip when ``upsample_sht=True``. Spherical
       attention itself only supports ``nlon_in % nlon_out == 0``, so the
       up-sample has to come first.
    2. :class:`NeighborhoodAttentionS2` at the output grid for local
       refinement (``in_chans -> in_chans``).
    3. Pointwise 1x1 projection ``in_chans -> out_chans``. Decoupling the
       channel projection from the attention keeps things well-defined when
       ``out_chans`` (e.g. 1 for precip) is not divisible by ``num_heads``.
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
    ):
        super().__init__()
        if in_chans % num_heads != 0:
            raise ValueError(
                f"AttentionDecoder: in_chans ({in_chans}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        if upsample_sht:
            sht = th.RealSHT(*inp_shape, grid=grid_in).float()
            isht = th.InverseRealSHT(
                *out_shape, lmax=sht.lmax, mmax=sht.mmax, grid=grid_out,
            ).float()
            self.upsample = nn.Sequential(sht, isht)
        else:
            self.upsample = th.ResampleS2(
                *inp_shape, *out_shape, grid_in=grid_in, grid_out=grid_out,
            )

        theta_cutoff = _neighborhood_radius_rad(out_shape[0], theta_cutoff_factor)
        self.attn = th.NeighborhoodAttentionS2(
            in_channels=in_chans,
            in_shape=out_shape,
            out_shape=out_shape,
            grid_in=grid_out,
            grid_out=grid_out,
            num_heads=num_heads,
            theta_cutoff=theta_cutoff,
            use_qknorm=qk_norm,
            bias=bias,
            out_channels=in_chans,
            optimized_kernel=attn_optimized_kernel,
        )

        self.project = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=bias)
        nn.init.normal_(self.project.weight, std=math.sqrt(2.0 / max(1, in_chans)))
        if bias:
            nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        with amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            x = self.upsample(x)
            x = self.attn(x)
        x = x.to(dtype)
        x = self.project(x)
        return x


# -----------------------------------------------------------------------------
# Positional embeddings
# -----------------------------------------------------------------------------


class _SpectralPositionEmbedding(nn.Module):
    """Spherical-harmonic positional embedding (channel-wise basis function).

    Each channel is one real spherical harmonic on the (h, w) grid, normalised
    to unit max amplitude. Cheap, deterministic, and well-suited to spherical
    transformer pretraining.

    Degree coverage (``max_degree``):

    - ``None`` (default): channel ``i`` is the ``i``-th harmonic in the
      standard ``(l, m)`` enumeration, so for ``num_chans`` channels the degree
      only reaches ``l ≈ ⌊√num_chans⌋``.  With ``num_chans=256`` that is just
      ``l≤15`` (spatial wavelength ~11°) — too low-frequency to let attention
      localise inside a small geodesic neighbourhood (→ blurry outputs).
    - integer: spread ``num_chans`` harmonics evenly (in enumeration index)
      over degrees ``0 … max_degree``, so the embedding carries high-frequency
      components (wavelength ~``180°/max_degree``) and can resolve fine
      position differences, while still including the smooth low-degree modes.
      ``max_degree`` is clamped to the grid bandlimit.
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
        isht = InverseRealSHT(nlat=H, nlon=W, grid=grid)

        # Build the list of (l, m) harmonics assigned to each channel.
        if max_degree is None:
            # Legacy: the first ``num_chans`` harmonics (all low-degree).
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
                idx = torch.linspace(0, len(all_pairs) - 1, num_chans).round().long().tolist()
                pairs = [all_pairs[j] for j in idx]
            else:
                # Fewer harmonics than channels (tiny grids): cycle through.
                pairs = [all_pairs[j % len(all_pairs)] for j in range(num_chans)]

        with torch.no_grad():
            pos_freq = torch.zeros(1, num_chans, isht.lmax, isht.mmax, dtype=torch.complex64)
            for i, (l, m) in enumerate(pairs):
                # Guard against out-of-range indices on small grids.
                if l >= isht.lmax or abs(m) >= isht.mmax:
                    continue
                if m < 0:
                    pos_freq[0, i, l, -m] = 1.0j
                else:
                    pos_freq[0, i, l, m] = 1.0
            pos_embed = isht(pos_freq)
            pos_embed = pos_embed / (pos_embed.abs().amax(dim=(-1, -2), keepdim=True) + 1.0e-8)
        self.register_buffer("position_embeddings", pos_embed.float(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embeddings


class _LearnableLatitudePositionEmbedding(nn.Module):
    """Per-latitude learnable bias (broadcast across longitudes).

    Lightweight alternative to the spectral embedding when you don't want to
    pay for an SHT at init time. ``(1, C, H, 1)`` parameter broadcast against
    every longitude.
    """

    def __init__(self, grid_shape: Tuple[int, int], num_chans: int):
        super().__init__()
        H = int(grid_shape[0])
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_chans, H, 1))
        nn.init.normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embeddings


def _build_pos_embedding(
    kind: str,
    grid_shape: Tuple[int, int],
    num_chans: int,
    grid: str,
) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "none":
        return nn.Identity()
    if kind == "spectral":
        return _SpectralPositionEmbedding(grid_shape, num_chans, grid)
    if kind in ("learnable_lat", "learnable"):
        return _LearnableLatitudePositionEmbedding(grid_shape, num_chans)
    raise ValueError(
        f"Unknown pos_embed='{kind}'. Supported: 'none', 'spectral', 'learnable_lat'."
    )


# -----------------------------------------------------------------------------
# Spherical transformer block
# -----------------------------------------------------------------------------


def _make_norm(kind: str, num_chans: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "none":
        return nn.Identity()
    if kind == "layer_norm":
        # LayerNorm over channels at every spatial location: rearrange via GroupNorm
        # with groups=1, which is equivalent and channel-first friendly.
        return nn.GroupNorm(num_groups=1, num_channels=num_chans, eps=1e-6, affine=True)
    if kind == "instance_norm":
        return nn.InstanceNorm2d(
            num_features=num_chans, eps=1e-6, affine=True, track_running_stats=False,
        )
    raise ValueError(
        f"Unknown normalization_layer='{kind}'. Supported: 'none', 'layer_norm', 'instance_norm'."
    )


class SphericalTransformerBlock(nn.Module):
    """Pre-norm spherical transformer block.

    Mirrors the channel-routing convention of LNO's ``NeuralOperatorBlock``:
    the spatial mixer (attention) keeps the channel count constant at
    ``in_chans``, and the MLP is the component that changes channels
    (``in_chans -> out_chans``). When ``in_chans != out_chans`` the MLP
    residual is disabled, matching LNO's ``block_skip="none"`` policy.

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
    ):
        super().__init__()

        if in_chans % num_heads != 0:
            raise ValueError(
                f"SphericalTransformerBlock: in_chans ({in_chans}) must be "
                f"divisible by num_heads ({num_heads}). When use_encoder=False, "
                f"the first block consumes in_channels directly, so pick a "
                f"num_heads that divides in_channels (e.g. 1, 2, 3, 6 for 78)."
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

        # Channel-changing MLP can't share the residual stream; mirror LNO's
        # block_skip="none" policy when in_chans != out_chans (see the comment
        # in `PrecipNeuralOperatorNet.__init__` about silently injecting
        # input channels as a bias if we instead cropped/identity-skipped).
        self._mlp_residual = self.in_chans == self.out_chans

    def _attn_forward(self, x: torch.Tensor) -> torch.Tensor:
        # NeighborhoodAttentionS2's optimized CUDA kernel only supports FP32 input,
        # so we disable autocast around it. AttentionS2 handles autocast natively.
        if self.attention_mode == "neighborhood":
            dtype = x.dtype
            with amp.autocast(device_type=x.device.type, enabled=False):
                return self.self_attn(x.float()).to(dtype)
        return self.self_attn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm0(x)
        h = self._attn_forward(h)
        x = residual + self.drop_path0(h)

        if self.mlp is not None:
            residual = x
            h = self.norm1(x)
            h = self.mlp(h)
            h = self.drop_path1(h)
            x = residual + h if self._mlp_residual else h
        return x


# -----------------------------------------------------------------------------
# PrecipAttentionNet
# -----------------------------------------------------------------------------


@dataclass
class _MetaData(ModelMetaData):
    name: str = "PrecipAttentionNet"
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
    """Fully attention-based precip diagnostic backbone.

    Drop-in replacement for :class:`PrecipNeuralOperatorNet`: same constructor
    kwargs for ``in_channels``, ``out_channels``, ``inp_shape``,
    ``model_grid_type``, ``sht_grid_type``, ``scale_factor``, ``embed_dim``,
    ``num_layers``, ``use_encoder``, ``use_decoder``, ``big_skip``,
    ``normalization_layer``, ``checkpointing_level``, etc.; same
    ``forward(x, noise=None)`` signature.

    The internal (processor) grid is either derived from ``scale_factor``
    (``h, w = inp_shape // scale_factor``) or set explicitly via
    ``latent_shape=(h, w)``, which takes precedence. The latent longitude
    ``w`` MUST divide the input longitude (spherical-attention p-shift);
    latitude ``h`` is free.

    ``use_encoder=False`` / ``use_decoder=False`` skip the spherical-attention
    encoder/decoder (no spatial up/downsampling), so the processor grid must
    equal the input grid (``scale_factor=1`` or ``latent_shape=inp_shape``). Channel routing in that case mirrors :class:`PrecipNeuralOperatorNet`:
    the first block's input channel count is ``in_channels`` (instead of
    ``embed_dim``) and/or the last block's output channel count is
    ``out_channels``; the channel change happens inside the block's MLP, and
    the MLP residual is disabled when ``in_chans != out_chans``. Because
    spherical attention requires ``num_heads | k_channels`` and
    ``num_heads | out_channels``, ``use_encoder=False`` constrains
    ``num_heads`` to divide ``in_channels`` (e.g. for 78 inputs: 1, 2, 3, 6).

    Attention-specific knobs:

    ``num_heads`` (int): number of attention heads. Must divide ``embed_dim``.
    ``attention_mode`` (str or list[str]): ``"neighborhood"`` (default,
    geodesic-cap local attention via ``NeighborhoodAttentionS2``) or
    ``"global"`` (full quadrature-weighted SDPA via ``AttentionS2``). A list
    of length ``num_layers`` mixes modes per block, e.g.
    ``["neighborhood", "global", "neighborhood", "global"]``.
    ``attn_theta_cutoff_factor`` (float): neighborhood radius in units of
    latitude rings on the processor grid. The radius in radians is
    ``factor * pi / (h - 1)`` where ``h`` is the internal latitude count.
    ``encoder_theta_cutoff_factor`` / ``decoder_theta_cutoff_factor`` set the
    same knob for the encoder (on the input grid) and decoder (on the output
    grid). Unrelated to any filter-basis: spherical attention is not a
    convolution and has no kernel basis.
    ``qk_norm`` (bool): apply RMSNorm to query/key projections (stabilises
    long-context attention).
    ``pos_embed`` (str): ``"none"`` / ``"spectral"`` / ``"learnable_lat"``.
    ``attn_drop_rate`` (float): dropout applied inside the softmax of the
    global attention path (no-op for the neighborhood path).
    ``attn_optimized_kernel`` (bool): try the CUDA neighborhood-attention
    kernel; falls back to the reference torch path if not built.

    The ``noise_mode`` knob of the conv-based net is accepted for API parity
    but raises if a noise input is supplied (no stochastic conditioning is
    wired in yet).
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
        **kwargs,
    ):
        super().__init__(meta=_MetaData())

        if n_history != 0:
            raise ValueError("PrecipAttentionNet currently only supports n_history=0.")
        if noise_mode is not None:
            raise NotImplementedError(
                "PrecipAttentionNet does not (yet) support noise conditioning; "
                "set noise_mode=None or use PrecipNeuralOperatorNet."
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

        if activation_function == "relu":
            act_layer = nn.ReLU
        elif activation_function == "gelu":
            act_layer = nn.GELU
        elif activation_function == "silu":
            act_layer = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # Internal (processor) spatial shape. Two ways to specify it:
        #   1. latent_shape=(h, w) — explicit, takes precedence.
        #   2. scale_factor — h,w = inp_shape // scale_factor.
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

        # Encoder/decoder mirror the LNO convention: when off they are pure
        # nn.Identity() and the channel routing in_channels -> embed_dim ->
        # out_channels is pushed into the first/last processor block (the
        # block's MLP changes channels; see SphericalTransformerBlock above).
        # The spatial shape can't change without the attention encoder/decoder
        # (NeighborhoodAttentionS2 is the only piece that up/downsamples), so
        # use_encoder=False / use_decoder=False require scale_factor=1.
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
                out_shape=self.inp_shape,
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
            if (self.h, self.w) != tuple(self.inp_shape):
                raise ValueError(
                    "use_decoder=False with PrecipAttentionNet requires the "
                    "processor grid to match the output grid (set scale_factor=1). "
                    f"Got out grid {self.inp_shape} but processor grid is "
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

        # Per-block channel routing mirrors PrecipNeuralOperatorNet (lno.py):
        #   first block:  in_ch = in_channels   if use_encoder=False else embed_dim
        #   last block:   out_ch = out_channels if use_decoder=False else embed_dim
        #   middle blocks always operate at embed_dim. The channel change
        #   happens inside SphericalTransformerBlock's MLP (the attention
        #   itself preserves channels at in_ch -> in_ch).
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
            pos_embed, (self.h, self.w), first_in_ch, sht_grid_type,
        )

        dpr = [float(x) for x in torch.linspace(0, path_drop_rate, max(1, self.num_layers))]
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
                )
                for i in range(self.num_layers)
            ]
        )

        if self.big_skip:
            self.residual_transform = nn.Conv2d(
                self.in_channels, self.out_channels, 1, bias=False,
            )
            scale = math.sqrt(0.5 / max(1, self.in_channels))
            nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.contiguous())

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def processor_blocks(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_embed(self.pos_drop(x))
        for blk in self.blocks:
            if self.checkpointing_level >= 3:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is not None:
            raise NotImplementedError(
                "PrecipAttentionNet does not (yet) support noise conditioning."
            )
        residual = x.contiguous() if self.big_skip else None

        if self.checkpointing_level >= 1:
            x = checkpoint(self.encode, x, use_reentrant=False)
        else:
            x = self.encode(x)

        x = self.processor_blocks(x)

        if self.checkpointing_level >= 1:
            x = checkpoint(self.decode, x, use_reentrant=False)
        else:
            x = self.decode(x)

        if self.big_skip:
            x = x + self.residual_transform(residual)
        return x
