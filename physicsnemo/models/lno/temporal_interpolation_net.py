# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spherical-attention temporal interpolation network.

Given two input fields at times t₀ and t₁, and a target fractional time
τ = (t_target − t₀) / (t₁ − t₀) ∈ (0, 1), this model produces the
interpolated field at t_target.

Architecture
------------

1. **Shared S2 encoder** (``AttentionEncoder`` from :mod:`attention_net`):
   maps each input frame ``(B, C_in, H, W)`` independently to a latent
   ``(B, D, h, w)``.  The encoded features are **clean content** — no
   positional or temporal embedding is baked into them.

2. **Position / time handling (DETR ``with_pos_embed`` convention)**: the
   embeddings are injected ONLY into the attention's query and key, never into
   the value or the residual.  The query gets ``PE + sinusoidal(τ)``; each key
   gets ``content + PE + sinusoidal(frame_time)`` (frame_time = 0 for t₀, 1 for
   t₁); the value is the clean encoded content.  All embeddings are fixed (no
   learnable parameters).  The carried feature stream therefore stays a pure
   signal — the embeddings steer *where* attention looks, not *what* it adds.

3. **Decoder** (one cross-attention + ``num_layers`` self-attention blocks):
   the latent content starts as the τ slot (from the query map) and is refined
   by

     a. a **single neighborhood cross-attention**
        (:class:`NeighborhoodTemporalCrossAttentionBlock`): the τ-slot latent
        (plus PE) queries each frame *locally* on the sphere via
        ``NeighborhoodAttentionS2``.  Each temporal step gets ``num_heads/2`` of
        the heads (no cross-frame QKV mixing); the two per-frame outputs are
        concatenated and projected back to ``D``.  The block itself has no query
        residual (clean values); the residual ``z = z + cross(z + PE, …)`` is
        added at the call site (decoder convention).
     b. a stack of ``num_layers`` **neighborhood self-attention**
        :class:`SphericalTransformerBlock` processor blocks.

   Local attention scales ~linearly with grid size, so the latent need not be
   downsampled (position is implicit in the geodesic neighborhood, plus the PE
   in the cross-attn query).

4. **S2 decoder** (``AttentionDecoder`` from :mod:`attention_net`):
   spherical up-sampling back to ``(H, W)`` followed by
   ``NeighborhoodAttentionS2`` refinement and a 1×1 channel projection to
   ``C_out``.  (A 1×1 conv when the latent grid equals the input grid.)

Usage
-----

.. code-block:: python

    model = TemporalInterpolationNet(
        in_channels=75,       # C per input frame (ERA5 channels)
        out_channels=1,       # precipitation target
        inp_shape=(181, 360),
        scale_factor=4,       # bottleneck at (45, 90)
        embed_dim=128,
        num_layers=2,
    )
    x   = torch.randn(2, 150, 181, 360)  # 2 × 75 channels
    tau = torch.tensor([0.5, 0.5])       # interpolate at half-time
    y   = model(x, tau)                  # (2, 1, 181, 360)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
import torch_harmonics as th
from torch.utils.checkpoint import checkpoint

from physicsnemo.core import ModelMetaData, Module

from .attention_net import (
    AttentionDecoder, AttentionEncoder, SphericalTransformerBlock,
    _SpectralPositionEmbedding, _neighborhood_radius_rad,
)


# ---------------------------------------------------------------------------
# Fixed sinusoidal time embedding (no parameters)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Fixed sinusoidal embedding of a scalar time value t ∈ [0, 1].

    Channel layout: ``[sin(2π·f₀·t), …, sin(2π·f_{D/2-1}·t),
                       cos(2π·f₀·t), …, cos(2π·f_{D/2-1}·t)]``
    with log-spaced frequencies ``fₖ`` ranging geometrically from ``1`` to
    ``max_freq``.  No learnable parameters.

    The frequencies are bounded (NOT ``2^k``): with ``embed_dim=256`` an octave
    schedule reaches ``2^127·2π ≈ 1e39`` which overflows fp32 to ``inf`` for any
    ``t≠0`` (``sin(inf)=NaN``).  A geometric span to ``max_freq`` keeps
    ``2π·f·t`` finite while still resolving fine differences in ``t``.
    """

    def __init__(self, embed_dim: int, max_freq: float = 1.0e3):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for sinusoidal embedding"
        half = embed_dim // 2
        if half == 1:
            freqs = torch.ones(1)
        else:
            # log-spaced 1 … max_freq (inclusive)
            freqs = torch.exp(torch.linspace(0.0, math.log(float(max_freq)), half))
        self.register_buffer("freqs", freqs.float(), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : ``(B,)`` or ``(N,)`` — time values.

        Returns
        -------
        ``(B, D)`` — fixed sinusoidal features.
        """
        angles = t.unsqueeze(-1) * self.freqs.unsqueeze(0) * (2.0 * math.pi)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# ---------------------------------------------------------------------------
# Spatiotemporal query map (no parameters)
# ---------------------------------------------------------------------------

class TemporalQueryMap(nn.Module):
    """Maps a scalar τ ∈ (0, 1) to a temporal feature map ``(B, D, h, w)``.

    Broadcasts :class:`SinusoidalTimeEmbedding` of τ over the spatial grid.
    The spatial positional embedding is **not** applied here — it lives on
    :attr:`TemporalInterpolationNet.spatial_embed` and is added externally
    (both to Q and to the encoded frames) so the same object is shared.
    """

    def __init__(self, embed_dim: int, h: int, w: int, **_):
        super().__init__()
        self.h, self.w = h, w
        self.time_embed = SinusoidalTimeEmbedding(embed_dim)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tau : ``(B,)``

        Returns
        -------
        ``(B, D, h, w)``
        """
        t = self.time_embed(tau)                                       # (B, D)
        return t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.h, self.w)


# ---------------------------------------------------------------------------
# Neighborhood cross-attention block  (per-frame heads, channel-concat output)
# ---------------------------------------------------------------------------

class NeighborhoodTemporalCrossAttentionBlock(nn.Module):
    """Per-frame neighborhood cross-attention on the sphere.

    A shared target-time query attends — *locally* on the sphere, via
    :class:`torch_harmonics.NeighborhoodAttentionS2` — against each of the two
    encoded frames **independently**.  Because the two frames are run through
    separate attention modules and their outputs are concatenated along the
    channel axis, the two temporal steps are routed onto disjoint heads:
    frame ``t₀`` occupies the first ``num_heads/2`` heads and frame ``t₁`` the
    last ``num_heads/2``, with no cross-frame mixing inside the QKV
    projections.  A 1×1 conv then projects the concatenated ``2·D`` channels
    back to ``D``.

    Why per-frame modules instead of one attention over a ``2·D``-channel K/V?
    ``NeighborhoodAttentionS2``'s Q/K/V projections are dense over *all* input
    channels, so concatenating the frames into one K/V and merely doubling
    ``num_heads`` would let the projection mix the two frames before the head
    split.  Running a module per frame keeps "half the heads per step" exact.

    **DETR-style position handling.**  The block follows the
    ``with_pos_embed`` convention (Carion et al., 2020): the spatial/temporal
    embeddings are mixed into the **query** and **key** only — i.e. they steer
    *where* attention looks — while the **value** is the clean encoded content
    and there is **no** additive residual from the query.  The carried
    (output) stream therefore stays a pure signal: embeddings influence the
    attention weights but are never accumulated into the feature values.

    Parameters
    ----------
    embed_dim:
        Per-frame feature width ``D``.  The query, each frame, and the output
        are all ``D`` channels; the internal concatenation is ``2·D``.
    num_heads:
        Total number of heads; **must be even**.  Each frame is attended with
        ``num_heads // 2`` heads, so ``embed_dim`` must be divisible by
        ``num_heads // 2``.
    h, w:
        Bottleneck grid dimensions.
    grid:
        torch-harmonics grid type of the bottleneck grid.
    theta_cutoff_factor:
        Neighborhood radius in units of latitude rings on the (h, w) grid
        (``radius = factor · π / (h − 1)``).
    qk_norm:
        Apply RMSNorm to Q/K inside the attention.
    bias:
        Add bias to the projections.
    attn_optimized_kernel:
        Use the optimized CUDA neighborhood-attention kernel when available.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        h: int,
        w: int,
        grid: str = "legendre-gauss",
        theta_cutoff_factor: float = 4.0,
        qk_norm: bool = False,
        bias: bool = True,
        attn_optimized_kernel: bool = True,
    ):
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError(
                f"NeighborhoodTemporalCrossAttentionBlock: num_heads "
                f"({num_heads}) must be even so the two temporal steps can be "
                f"split half/half across the heads."
            )
        heads_per_frame = num_heads // 2
        if embed_dim % heads_per_frame != 0:
            raise ValueError(
                f"NeighborhoodTemporalCrossAttentionBlock: embed_dim "
                f"({embed_dim}) must be divisible by num_heads//2 "
                f"({heads_per_frame})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.h, self.w = h, w

        theta_cutoff = _neighborhood_radius_rad(h, theta_cutoff_factor)
        attn_kwargs = dict(
            in_channels=embed_dim,
            in_shape=(h, w),
            out_shape=(h, w),
            grid_in=grid,
            grid_out=grid,
            num_heads=heads_per_frame,
            theta_cutoff=theta_cutoff,
            use_qknorm=qk_norm,
            bias=bias,
            out_channels=embed_dim,
            optimized_kernel=attn_optimized_kernel,
        )
        # Distinct weights per frame → truly disjoint heads.  The frame
        # identity is already tagged onto z0/z1 via the fixed time embeddings.
        self.attn0 = th.NeighborhoodAttentionS2(**attn_kwargs)
        self.attn1 = th.NeighborhoodAttentionS2(**attn_kwargs)

        self.proj = nn.Conv2d(2 * embed_dim, embed_dim, kernel_size=1, bias=bias)
        nn.init.normal_(self.proj.weight, std=math.sqrt(1.0 / max(1, 2 * embed_dim)))
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key0: torch.Tensor,
        value0: torch.Tensor,
        key1: torch.Tensor,
        value1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query  : ``(B, D, h, w)`` — target-time query (PE + τ embedding).
        key0   : ``(B, D, h, w)`` — frame t₀ content + PE + frame-time tag.
        value0 : ``(B, D, h, w)`` — frame t₀ **clean** content (no embeddings).
        key1   : ``(B, D, h, w)`` — frame t₁ content + PE + frame-time tag.
        value1 : ``(B, D, h, w)`` — frame t₁ **clean** content (no embeddings).

        Returns
        -------
        ``(B, D, h, w)`` — clean attended content (no additive embeddings).
        """
        dtype = query.dtype
        # NeighborhoodAttentionS2's optimized CUDA kernel is fp32-only.
        with amp.autocast(device_type=query.device.type, enabled=False):
            qf = query.float()
            o0 = self.attn0(qf, key=key0.float(), value=value0.float())  # step t₀
            o1 = self.attn1(qf, key=key1.float(), value=value1.float())  # step t₁
        out = torch.cat([o0, o1], dim=1).to(dtype)    # (B, 2D, h, w)
        out = self.proj(out)                           # (B, D, h, w)
        # No residual from query: keep the carried stream a clean signal —
        # the embeddings only shaped the attention weights (via Q/K), never
        # the values.
        return out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@dataclass
class _MetaData(ModelMetaData):
    name: str = "TemporalInterpolationNet"
    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = True
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class TemporalInterpolationNet(Module):
    """Spherical-attention temporal interpolation network.

    See module docstring for the full architectural description.

    Parameters
    ----------
    in_channels:
        Number of channels in **one** input frame.  The model input ``x``
        must have shape ``(B, 2 * in_channels, H, W)`` (two frames
        concatenated along channels).
    out_channels:
        Number of output channels.
    inp_shape:
        ``(H, W)`` of the input/output grid (equiangular).
    scale_factor:
        Convenience fallback for deriving the bottleneck grid when
        ``latent_shape`` is not given: ``h ≈ H / scale_factor`` and ``w`` is the
        largest divisor of ``W`` that is ≤ ``W / scale_factor``.  Use
        ``scale_factor=1`` to process at full resolution.
    latent_shape:
        Explicit bottleneck grid ``(h, w)``.  Takes precedence over
        ``scale_factor``.  ``w`` MUST divide ``W`` (spherical-attention p-shift);
        ``h`` may be any value.  Set equal to ``inp_shape`` to skip the
        encoder/decoder and process at full resolution.
    model_grid_type:
        Grid type of the input/output grid (passed to ``AttentionEncoder``/
        ``AttentionDecoder``).
    sht_grid_type:
        Grid type of the bottleneck (internal) grid.
    embed_dim:
        Latent feature width ``D``.
    num_layers:
        Number of neighborhood S2 self-attention processor blocks that follow
        the single cross-attention block. Default 1.
    num_heads:
        Number of attention heads.  Must divide ``embed_dim`` for the
        processor/encoder/decoder, and must be **even** for the cross-attention
        (each temporal step uses ``num_heads/2`` heads).
    encoder_theta_cutoff_factor:
        Neighborhood radius for the S2 encoder block.
    decoder_theta_cutoff_factor:
        Neighborhood radius for the S2 decoder block.
    cross_theta_cutoff_factor:
        Neighborhood radius (in latitude rings on the bottleneck grid) for the
        cross-attention block.
    processor_theta_cutoff_factor:
        Neighborhood radius (in latitude rings on the bottleneck grid) for the
        self-attention processor blocks.
    pos_embed_max_degree:
        Max spherical-harmonic degree for the spatial positional embedding.
        ``None`` keeps only the lowest ``≈⌊√embed_dim⌋`` degrees (smooth, poor
        at localising attention).  An integer spreads ``embed_dim`` harmonics up
        to that degree so the embedding carries high spatial frequencies
        (wavelength ~``180°/max_degree``) and the attention can localise —
        recommended to be ≳ ``180 / neighborhood_radius_deg`` (e.g. ~64–90 for
        a ~2–4° cap).
    mlp_ratio:
        Hidden-size multiplier inside each ``SphericalTransformerBlock`` MLP.
    path_drop_rate:
        Stochastic-depth drop rate (applied uniformly to all processor blocks).
    checkpointing_level:
        ``0`` — no checkpointing.  ``1`` — checkpoint encoder/decoder.
        ``3`` — also checkpoint every processor block.
    upsample_sht:
        Use band-limited SHT upsampling in the decoder instead of bilinear.
        Only meaningful when ``scale_factor > 1``.
    qk_norm:
        Apply RMSNorm to Q/K inside the S2 attention blocks.
    bias:
        Enable bias in linear projections.
    """

    @staticmethod
    def _largest_lon_divisor(W: int, scale_factor: int) -> int:
        """Largest divisor of ``W`` that is ≤ ``W / scale_factor``.

        The spherical-attention p-shift requires ``W % w == 0``. We pick the
        coarsest bottleneck width that still divides ``W`` so the downsample
        is as close as possible to the requested ``scale_factor`` without
        violating the kernel constraint.
        """
        target = max(1, W // scale_factor)
        for w in range(target, 0, -1):
            if W % w == 0:
                return w
        return 1

    def __init__(
        self,
        in_channels: int = 75,
        out_channels: int = 1,
        inp_shape: Tuple[int, int] = (181, 360),
        scale_factor: int = 4,
        latent_shape: Optional[Tuple[int, int]] = None,
        model_grid_type: str = "equiangular",
        sht_grid_type: str = "legendre-gauss",
        embed_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 8,
        encoder_theta_cutoff_factor: float = 4.0,
        decoder_theta_cutoff_factor: float = 4.0,
        cross_theta_cutoff_factor: float = 4.0,
        processor_theta_cutoff_factor: float = 4.0,
        pos_embed_max_degree: Optional[int] = None,
        mlp_ratio: float = 2.0,
        path_drop_rate: float = 0.0,
        checkpointing_level: int = 0,
        upsample_sht: bool = False,
        qk_norm: bool = False,
        bias: bool = True,
        attn_optimized_kernel: bool = True,
        **kwargs,
    ):
        super().__init__(meta=_MetaData())

        H, W = int(inp_shape[0]), int(inp_shape[1])
        # The bottleneck (latent) grid can be specified two ways:
        #   1. latent_shape=(h, w) — explicit, takes precedence.
        #   2. scale_factor — h,w derived from inp_shape.
        # Latitude (h) may be arbitrary; longitude (w), however, MUST divide W
        # exactly (nlon_in % nlon_out == 0), a hard requirement of the
        # spherical-attention p-shift kernel.
        if latent_shape is not None:
            h, w = int(latent_shape[0]), int(latent_shape[1])
        elif scale_factor > 1:
            h = max(1, round(H / scale_factor))
            w = self._largest_lon_divisor(W, scale_factor)
        else:
            h, w = H, W
        

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.inp_shape = (H, W)
        self.scale_factor = scale_factor
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.checkpointing_level = int(checkpointing_level)
        self.h, self.w = h, w
        # Whether the latent grid differs from the input grid (→ use the
        # spherical encoder/decoder). Equal grids skip resampling entirely.
        self._downsamples = (h, w) != (H, W)
        # Grid type the latent actually lives on. When we downsample, the
        # encoder resamples the (model_grid_type) input onto the internal
        # (sht_grid_type) latent. When we DON'T downsample, the latent is the
        # input grid itself, so all internal attention must use model_grid_type
        # — otherwise the quadrature weights / latitudes would be computed for
        # the wrong grid.
        self._latent_grid = sht_grid_type if self._downsamples else model_grid_type

        # ----------------------------------------------------------------
        # 1. Shared spatial encoder (applied once per input frame)
        # ----------------------------------------------------------------
        if self._downsamples:
            self.encoder = AttentionEncoder(
                inp_shape=(H, W),
                out_shape=(h, w),
                in_chans=self.in_channels,
                out_chans=embed_dim,
                grid_in=model_grid_type,
                grid_out=sht_grid_type,
                num_heads=num_heads,
                theta_cutoff_factor=encoder_theta_cutoff_factor,
                qk_norm=qk_norm,
                bias=bias,
                attn_optimized_kernel=attn_optimized_kernel,
            )
        else:
            # latent grid == input grid: encoder is a plain 1x1 lift.
            self.encoder = nn.Conv2d(self.in_channels, embed_dim, kernel_size=1, bias=bias)

        # ----------------------------------------------------------------
        # 2. Fixed sinusoidal frame embeddings (no parameters)
        #    Frame t₀ → sinusoidal(0.0),  frame t₁ → sinusoidal(1.0).
        #    Same coordinate system as the query map so the cross-attention
        #    can compare them directly.
        # ----------------------------------------------------------------
        _time_embed = SinusoidalTimeEmbedding(embed_dim)
        frame_embeds = _time_embed(torch.tensor([0.0, 1.0]))  # (2, D)
        self.register_buffer("frame_embeds", frame_embeds, persistent=False)

        # ----------------------------------------------------------------
        # 3. Shared spatial positional embedding — injected BEFORE the
        #    encoder's self-attention (into the encoder's pos_embed slot)
        #    AND added to the query map, so K/V and Q share the same
        #    spatial coordinate system throughout.
        # ----------------------------------------------------------------
        self.spatial_embed = _SpectralPositionEmbedding(
            (h, w), embed_dim, self._latent_grid, max_degree=pos_embed_max_degree,
        )
        if self._downsamples:
            # Attach to the encoder so PE is added after resample, before attn.
            self.encoder.pos_embed = self.spatial_embed

        # ----------------------------------------------------------------
        # 4. Target-time query map  τ → (B, D, h, w)  (no parameters)
        # ----------------------------------------------------------------
        self.query_map = TemporalQueryMap(embed_dim, h, w, grid=self._latent_grid)

        # ----------------------------------------------------------------
        # 5. Neighborhood cross-attention.  A shared target-time query
        #    attends locally on the sphere against each frame independently;
        #    the two temporal steps live on disjoint heads (half each) and
        #    are concatenated along channels then projected back to D.
        # ----------------------------------------------------------------
        # SINGLE cross-attention block: the target-time query attends once to
        # the two input frames to pull their content into the latent, then the
        # self-attention processor stack refines it. (No per-layer cross-attn —
        # that interleaved variant tripled the heaviest ops and, with no
        # residual normalization, accumulated magnitude across depth → OOM +
        # divergence.)
        self.cross_attn = NeighborhoodTemporalCrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            h=h,
            w=w,
            grid=self._latent_grid,
            theta_cutoff_factor=cross_theta_cutoff_factor,
            qk_norm=qk_norm,
            bias=bias,
            attn_optimized_kernel=attn_optimized_kernel,
        )

        # ----------------------------------------------------------------
        # 6. Processor blocks — neighborhood S2 self-attention on the latent
        #    grid. Local (geodesic-cap) attention scales ~linearly with grid
        #    size, so the latent need not be downsampled at all to stay
        #    tractable (unlike global AttentionS2 whose cost is O((h·w)²)).
        # ----------------------------------------------------------------
        processor_theta_cutoff = _neighborhood_radius_rad(
            h, processor_theta_cutoff_factor
        )
        dpr = [float(x) for x in torch.linspace(0, path_drop_rate, max(1, num_layers))]
        self.blocks = nn.ModuleList([
            SphericalTransformerBlock(
                in_shape=(h, w),
                grid=self._latent_grid,
                in_chans=embed_dim,
                out_chans=embed_dim,
                num_heads=num_heads,
                attention_mode="neighborhood",
                attn_theta_cutoff=processor_theta_cutoff,
                mlp_ratio=mlp_ratio,
                path_drop_rate=dpr[i],
                normalization_layer="none",
                bias=bias,
                use_mlp=True,
                qk_norm=qk_norm,
                attn_optimized_kernel=attn_optimized_kernel,
                checkpointing_level=checkpointing_level,
            )
            for i in range(num_layers)
        ])

        # ----------------------------------------------------------------
        # 7. Spatial decoder
        # ----------------------------------------------------------------
        if self._downsamples:
            self.decoder = AttentionDecoder(
                inp_shape=(h, w),
                out_shape=(H, W),
                in_chans=embed_dim,
                out_chans=out_channels,
                grid_in=sht_grid_type,
                grid_out=model_grid_type,
                num_heads=num_heads,
                theta_cutoff_factor=decoder_theta_cutoff_factor,
                qk_norm=qk_norm,
                bias=bias,
                attn_optimized_kernel=attn_optimized_kernel,
                upsample_sht=upsample_sht,
            )
        else:
            # scale_factor=1: plain output projection
            self.decoder = nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=bias)

    # ------------------------------------------------------------------
    # Sub-routines (kept as named methods for gradient checkpointing)
    # ------------------------------------------------------------------

    def _encode_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one frame into **clean** latent content (no embeddings).

        Spatial/temporal embeddings are deliberately NOT added to the carried
        feature stream here — following the DETR ``with_pos_embed`` convention
        they are injected only into the cross-attention Q/K (see ``forward``),
        so the values stay a pure signal.  In the downsampling case the encoder
        still uses ``spatial_embed`` internally to make its own cross-resolution
        attention position-aware, but that PE goes into the encoder's attention
        query only, not into the returned content.
        """
        return self.encoder(x)                                 # (B, D, h, w)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, tau: torch.Tensor, noise=None) -> torch.Tensor:
        # noise accepted for API parity with DiagnosticModelWrapper but unused.
        """
        Parameters
        ----------
        x : ``(B, 2, C, H, W)``  — two input frames along the temporal axis.
            Frame t₀ is ``x[:, 0]``, frame t₁ is ``x[:, 1]``.
        tau : ``(B,)``
            Fractional interpolation time τ = (t_target − t₀) / (t₁ − t₀)
            in the open interval ``(0, 1)``.

        Returns
        -------
        ``(B, out_channels, H, W)``

        """

        if x.dim() == 4:
            # Legacy flat layout (B, 2*C, H, W) → reshape to (B, 2, C, H, W).
            B, _, H, W = x.shape
            x = x.view(B, 2, self.in_channels, H, W)
        x0, x1 = x[:, 0], x[:, 1]   # (B, C, H, W) each

        # 1. Encode each frame into CLEAN content (shared weights, no embeds).
        if self.checkpointing_level >= 1:
            z0 = checkpoint(self._encode_frame, x0, use_reentrant=False)
            z1 = checkpoint(self._encode_frame, x1, use_reentrant=False)
        else:
            z0 = self._encode_frame(x0)
            z1 = self._encode_frame(x1)

        # 2. Assemble the embeddings used ONLY inside the attention (DETR
        #    with_pos_embed style): the spatial PE is shared by Q and K; each
        #    key carries its frame-time tag. The frames (keys/values) are fixed
        #    across layers, so build them once. Values stay CLEAN.
        tau = tau.to(dtype=x.dtype, device=x.device)
        pe = self.spatial_embed.position_embeddings.to(z0.dtype)   # (1, D, h, w)
        fe0 = self.frame_embeds[0].to(z0.dtype).view(1, -1, 1, 1)
        fe1 = self.frame_embeds[1].to(z0.dtype).view(1, -1, 1, 1)
        key0, key1 = z0 + pe + fe0, z1 + pe + fe1                  # content + PE + time

        # 3. Decoder: ONE cross-attention pulls the two frames' content into the
        #    τ slot (residual onto the query; PE added only to form the query,
        #    never carried in the content so values stay clean), then the
        #    self-attention processor stack refines the latent.
        z = self.query_map(tau)                          # (B, D, h, w), τ slot
        if self.checkpointing_level >= 3:
            z = z + checkpoint(self.cross_attn, z + pe, key0, z0, key1, z1, use_reentrant=False)
        else:
            z = z + self.cross_attn(z + pe, key0, z0, key1, z1)   # single cross-attn
        for blk in self.blocks:
            if self.checkpointing_level >= 3:
                z = checkpoint(blk, z, use_reentrant=False)
            else:
                z = blk(z)                                  # self-attn

        # 4. Decode to output grid
        if self.checkpointing_level >= 1:
            out = checkpoint(self._decode, z, use_reentrant=False)
        else:
            out = self._decode(z)
        return out
