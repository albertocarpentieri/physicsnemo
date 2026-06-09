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
   maps each input frame ``(B, C_in, H, W)`` independently to a lower-
   resolution latent ``(B, D, h, w)`` via a 1×1 lift followed by
   ``NeighborhoodAttentionS2`` downsampling.

2. **Frame temporal embeddings**: a learnable ``(D,)`` bias for each of the
   two input frames, added to the encoded representation before the context
   is formed.

3. **Context formation**: the two temporally-tagged latent maps are
   concatenated along the channel axis → ``(B, 2D, h, w)``.

4. **Target-time query map**: sinusoidal Fourier features of the scalar τ are
   projected by a 2-layer MLP to ``(B, D)`` and then broadcast to a spatial
   map ``(B, D, h, w)``, to which a learnable spatial bias is added.

5. **S2 cross-attention block**: flattens Q (from the query map) and K/V
   (from the context) to sequence form, runs multi-head attention, and
   reshapes back to ``(B, D, h, w)``.  A residual path adds the query map
   to the cross-attention output.

6. **Processor blocks**: ``num_layers`` :class:`SphericalTransformerBlock`
   self-attention layers at the bottleneck resolution.

7. **S2 decoder** (``AttentionDecoder`` from :mod:`attention_net`):
   spherical up-sampling back to ``(H, W)`` followed by
   ``NeighborhoodAttentionS2`` refinement and a 1×1 channel projection to
   ``C_out``.

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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_harmonics.quadrature import precompute_latitudes

from physicsnemo.core import ModelMetaData, Module

from .attention_net import AttentionDecoder, AttentionEncoder, SphericalTransformerBlock
from ._layers import DropPath


# ---------------------------------------------------------------------------
# Fourier-feature target-time query map
# ---------------------------------------------------------------------------

class TemporalQueryMap(nn.Module):
    """Maps a scalar τ ∈ (0, 1) to a spatial feature map ``(B, D, h, w)``.

    A bank of sinusoidal Fourier features of τ is projected by a 2-layer MLP
    to ``embed_dim`` values, which are broadcast spatially and added to a
    learnable spatial bias.  This gives every spatial token a time-conditional
    query embedding that is also aware of its position on the sphere.

    Parameters
    ----------
    embed_dim:
        Feature dimension ``D`` of the output map.
    h, w:
        Spatial dimensions of the bottleneck grid.
    n_fourier:
        Number of sinusoidal feature pairs (total feature size = ``n_fourier``).
        Must be even.
    """

    def __init__(self, embed_dim: int, h: int, w: int, n_fourier: int = 32):
        super().__init__()
        assert n_fourier % 2 == 0, "n_fourier must be even"
        half = n_fourier // 2
        # Log-spaced frequencies: 2^0, 2^1, … for a wide bandwidth
        freqs = (2.0 ** torch.arange(half)).float()
        self.register_buffer("freqs", freqs, persistent=False)  # (half,)

        self.mlp = nn.Sequential(
            nn.Linear(n_fourier, embed_dim, bias=True),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )
        # Learnable per-location additive spatial bias
        self.spatial_bias = nn.Parameter(torch.zeros(1, embed_dim, h, w))
        nn.init.trunc_normal_(self.spatial_bias, std=0.02)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tau : ``(B,)`` — fractional time in ``(0, 1)``.

        Returns
        -------
        ``(B, D, h, w)``
        """
        # Fourier embedding: (B, n_fourier)
        angles = tau.unsqueeze(-1) * self.freqs.unsqueeze(0) * (2.0 * math.pi)
        feat = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # MLP → (B, D) → (B, D, 1, 1) → broadcast
        d_vec = self.mlp(feat).unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)
        return d_vec + self.spatial_bias                      # (B, D, h, w)


# ---------------------------------------------------------------------------
# Spherical cross-attention block  (2-frame KV with duplicated quad weights)
# ---------------------------------------------------------------------------

class SphericalTemporalCrossAttentionBlock(nn.Module):
    """Cross-attend a target-time query against a 2-frame context on the sphere.

    Follows the same quadrature-weighted attention formulation as
    ``torch_harmonics.AttentionS2`` (Bonev et al., NeurIPS 2025).

    Q comes from the target-time query map ``(B, D, h, w)``.
    K and V come from the two encoded input frames, each ``(B, D, h, w)``.
    The full KV token sequence has length ``2·h·w``: frame 0 tokens first,
    then frame 1 tokens.

    **Quadrature weights**: every token at latitude ``lat_i`` represents a
    surface-area element ``Ω_{ij} = (2π / W) · w_i`` on the sphere.  The
    same grid is shared by both frames, so the area weights are simply
    repeated twice to cover all ``2·h·w`` KV tokens::

        log_w = [log Ω_{0,0}, ..., log Ω_{h-1,W-1},   ← frame 0
                 log Ω_{0,0}, ..., log Ω_{h-1,W-1}]   ← frame 1

    These are passed as an additive ``attn_mask`` to
    ``F.scaled_dot_product_attention`` exactly as in ``AttentionS2``, so the
    softmax denominator becomes a properly area-weighted integral over the
    sphere sampled twice — once per temporal frame.

    A pre-norm on Q and a post-norm on the output, with a residual from Q,
    follow the standard transformer pre-norm convention.

    Parameters
    ----------
    embed_dim:
        Feature width ``D``; both Q and each frame's features must be ``D``.
    num_heads:
        Number of attention heads; must divide ``embed_dim``.
    h, w:
        Spatial dimensions of the bottleneck grid.
    grid:
        torch-harmonics grid type used to compute quadrature weights
        (e.g. ``"legendre-gauss"`` or ``"equiangular"``).
    bias:
        Add bias to the Q/K/V/output projections.
    dropout:
        Dropout on attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        h: int,
        w: int,
        grid: str = "legendre-gauss",
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"SphericalTemporalCrossAttentionBlock: embed_dim ({embed_dim}) "
                f"must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # 1×1 conv projections — matching the AttentionS2 convention so that
        # the spatial layout (B, C, H, W) is never transposed until needed.
        scale_qk = math.sqrt(6.0 / (2 * embed_dim))
        scale_v  = math.sqrt(6.0 / (2 * embed_dim))
        scale_o  = math.sqrt(3.0 / embed_dim)

        self.q_proj = nn.Conv2d(embed_dim,     embed_dim, 1, bias=bias)
        self.k_proj = nn.Conv2d(embed_dim,     embed_dim, 1, bias=bias)
        self.v_proj = nn.Conv2d(embed_dim,     embed_dim, 1, bias=bias)
        self.out_proj = nn.Conv2d(embed_dim,   embed_dim, 1, bias=bias)

        nn.init.uniform_(self.q_proj.weight,   -scale_qk, scale_qk)
        nn.init.uniform_(self.k_proj.weight,   -scale_qk, scale_qk)
        nn.init.uniform_(self.v_proj.weight,   -scale_v,  scale_v)
        nn.init.uniform_(self.out_proj.weight, -scale_o,  scale_o)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        self.norm_q = nn.GroupNorm(1, embed_dim, eps=1e-6)
        self.norm_out = nn.GroupNorm(1, embed_dim, eps=1e-6)

        # ---- Spherical quadrature weights, duplicated for 2 frames ----
        # precompute_latitudes returns (colatitudes, weights) where weights
        # integrate over the colatitude axis.  Multiplying by 2π/W gives the
        # solid-angle area of each (lat, lon) cell on the unit sphere.
        _, wgl = precompute_latitudes(h, grid=grid)          # (h,), sum ≈ 2
        quad_weights = (2.0 * math.pi / w) * wgl.float()    # (h,)
        quad_weights = quad_weights.unsqueeze(1).expand(h, w).reshape(h * w)  # (N,)

        # Duplicate: KV tokens = frame0 tokens + frame1 tokens on the same sphere.
        log_w_2 = torch.log(quad_weights.clamp(min=1e-8)).repeat(2)  # (2N,)
        # Shape (1, 1, 1, 2N) broadcasts to (B, heads, N_q, 2N) in SDPA.
        self.register_buffer(
            "log_quad_weights_2",
            log_w_2.reshape(1, 1, 1, -1),
            persistent=False,
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        query   : ``(B, D, h, w)``  — target-time query map
        context : ``(B, 2D, h, w)`` — two encoded frames concatenated along
                  the channel axis (frame 0: channels ``0:D``, frame 1: ``D:2D``)

        Returns
        -------
        ``(B, D, h, w)``
        """
        B, D, h, w = query.shape
        N = h * w
        H = self.num_heads
        d = self.head_dim

        # Split context into individual frame tensors
        ctx0 = context[:, :D]   # (B, D, h, w)
        ctx1 = context[:, D:]   # (B, D, h, w)

        # Pre-norm on query (in channel-first layout)
        q_normed = self.norm_q(query)                       # (B, D, h, w)

        # Q / K / V projections  (1×1 conv, channel-first)
        Q = self.q_proj(q_normed)                           # (B, D, h, w)
        K0 = self.k_proj(ctx0)                              # (B, D, h, w)
        K1 = self.k_proj(ctx1)                              # (B, D, h, w)  shared weights
        V0 = self.v_proj(ctx0)                              # (B, D, h, w)
        V1 = self.v_proj(ctx1)                              # (B, D, h, w)  shared weights

        # Reshape → (B, heads, N, head_dim)
        def _to_heads(t):
            return t.reshape(B, H, d, N).permute(0, 1, 3, 2)   # (B, H, N, d)

        Q_h = _to_heads(Q)                                  # (B, H, N, d)
        # Concatenate frame 0 and frame 1 tokens along the spatial axis
        K_h = torch.cat([_to_heads(K0), _to_heads(K1)], dim=2)  # (B, H, 2N, d)
        V_h = torch.cat([_to_heads(V0), _to_heads(V1)], dim=2)  # (B, H, 2N, d)

        # Quadrature-weighted cross-attention.
        # log_quad_weights_2 has shape (1, 1, 1, 2N) and is added to the raw
        # attention logits before softmax — identical to AttentionS2's usage.
        attn_out = F.scaled_dot_product_attention(
            Q_h, K_h, V_h,
            attn_mask=self.log_quad_weights_2.to(Q_h.dtype),
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, N, d)

        # Re-assemble spatial layout: (B, H, N, d) → (B, D, h, w)
        out = attn_out.permute(0, 1, 3, 2).reshape(B, D, h, w)
        out = self.out_proj(out)                            # (B, D, h, w)
        out = self.norm_out(out)

        # Residual from (unnormalised) query
        return out + query


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
        Spatial downsampling factor for the bottleneck.  Both ``H`` and ``W``
        must be divisible by ``scale_factor``.  Use ``scale_factor=1`` to
        skip the encoder/decoder and process at full resolution.
    model_grid_type:
        Grid type of the input/output grid (passed to ``AttentionEncoder``/
        ``AttentionDecoder``).
    sht_grid_type:
        Grid type of the bottleneck (internal) grid.
    embed_dim:
        Latent feature width ``D``.
    num_layers:
        Number of :class:`SphericalTransformerBlock` processor layers.
    num_heads:
        Number of attention heads (must divide ``embed_dim``).
    attn_theta_cutoff_factor:
        Neighborhood radius for processor blocks, in units of latitude
        rings on the bottleneck grid.
    encoder_theta_cutoff_factor:
        Neighborhood radius for the S2 encoder block.
    decoder_theta_cutoff_factor:
        Neighborhood radius for the S2 decoder block.
    n_fourier:
        Number of sinusoidal feature pairs in the temporal query map.
    mlp_ratio:
        Hidden-size multiplier inside each ``SphericalTransformerBlock`` MLP.
    normalization_layer:
        ``"layer_norm"`` or ``"none"``; applied inside processor blocks.
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

    def __init__(
        self,
        in_channels: int = 75,
        out_channels: int = 1,
        inp_shape: Tuple[int, int] = (181, 360),
        scale_factor: int = 4,
        model_grid_type: str = "equiangular",
        sht_grid_type: str = "legendre-gauss",
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        attn_theta_cutoff_factor: float = 4.0,
        encoder_theta_cutoff_factor: float = 4.0,
        decoder_theta_cutoff_factor: float = 4.0,
        n_fourier: int = 32,
        mlp_ratio: float = 2.0,
        normalization_layer: str = "layer_norm",
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
        h = H // scale_factor
        w = W // scale_factor

        if scale_factor > 1:
            if H % scale_factor != 0 or W % scale_factor != 0:
                raise ValueError(
                    f"TemporalInterpolationNet: inp_shape {(H, W)} must be "
                    f"divisible by scale_factor={scale_factor}."
                )
            if W % w != 0:
                raise ValueError(
                    f"TemporalInterpolationNet: nlon_in ({W}) must be a "
                    f"multiple of nlon_out ({w}) for the spherical attention "
                    f"p-shift."
                )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.inp_shape = (H, W)
        self.scale_factor = scale_factor
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.checkpointing_level = int(checkpointing_level)
        self.h, self.w = h, w

        # ----------------------------------------------------------------
        # 1. Shared spatial encoder (applied once per input frame)
        # ----------------------------------------------------------------
        if scale_factor > 1:
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
            # scale_factor=1: encoder is a plain 1x1 lift (no spatial change)
            self.encoder = nn.Conv2d(self.in_channels, embed_dim, kernel_size=1, bias=bias)

        # ----------------------------------------------------------------
        # 2. Frame temporal embeddings — learnable (D,) per frame index
        # ----------------------------------------------------------------
        self.frame_embed = nn.Embedding(2, embed_dim)
        nn.init.trunc_normal_(self.frame_embed.weight, std=0.02)

        # ----------------------------------------------------------------
        # 3. Target-time query map  τ → (B, D, h, w)
        # ----------------------------------------------------------------
        self.query_map = TemporalQueryMap(embed_dim, h, w, n_fourier=n_fourier)

        # ----------------------------------------------------------------
        # 4. Spherical cross-attention  Q: query map,  KV: 2-frame context
        #    Quadrature weights duplicated for the 2·h·w KV token sequence.
        # ----------------------------------------------------------------
        self.cross_attn = SphericalTemporalCrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            h=h,
            w=w,
            grid=sht_grid_type,
            bias=bias,
        )
        # GroupNorm on the concatenated 2-frame context before cross-attn.
        self.context_norm = nn.GroupNorm(1, 2 * embed_dim, eps=1e-6)

        # ----------------------------------------------------------------
        # 5. Processor blocks (SphericalTransformerBlocks at bottleneck)
        # ----------------------------------------------------------------
        from .attention_net import _neighborhood_radius_rad
        attn_cutoff = _neighborhood_radius_rad(h, attn_theta_cutoff_factor)

        dpr = [float(x) for x in torch.linspace(0, path_drop_rate, max(1, num_layers))]
        self.blocks = nn.ModuleList([
            SphericalTransformerBlock(
                in_shape=(h, w),
                grid=sht_grid_type,
                in_chans=embed_dim,
                out_chans=embed_dim,
                num_heads=num_heads,
                attention_mode="neighborhood",
                attn_theta_cutoff=attn_cutoff,
                mlp_ratio=mlp_ratio,
                path_drop_rate=dpr[i],
                normalization_layer=normalization_layer,
                bias=bias,
                use_mlp=True,
                qk_norm=qk_norm,
                attn_optimized_kernel=attn_optimized_kernel,
                checkpointing_level=checkpointing_level,
            )
            for i in range(num_layers)
        ])

        # ----------------------------------------------------------------
        # 6. Spatial decoder
        # ----------------------------------------------------------------
        if scale_factor > 1:
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

    def _encode_frame(self, x: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """Encode one frame and add its temporal embedding."""
        z = self.encoder(x)                                    # (B, D, h, w)
        e = self.frame_embed.weight[frame_idx].view(1, -1, 1, 1)
        return z + e

    def _build_context(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        return torch.cat([z0, z1], dim=1)                      # (B, 2D, h, w)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : ``(B, 2 * in_channels, H, W)``
            Two consecutive input frames concatenated along the channel axis.
            The first ``in_channels`` channels are frame t₀; the next
            ``in_channels`` channels are frame t₁.
        tau : ``(B,)``
            Fractional interpolation time τ = (t_target − t₀) / (t₁ − t₀)
            in the open interval ``(0, 1)``.

        Returns
        -------
        ``(B, out_channels, H, W)``
        """
        C = self.in_channels
        x0, x1 = x[:, :C], x[:, C:2 * C]   # split the two frames

        # 1. Encode each frame (shared weights, independent passes)
        if self.checkpointing_level >= 1:
            z0 = checkpoint(self._encode_frame, x0, 0, use_reentrant=False)
            z1 = checkpoint(self._encode_frame, x1, 1, use_reentrant=False)
        else:
            z0 = self._encode_frame(x0, 0)
            z1 = self._encode_frame(x1, 1)

        # 2. Form context: (B, 2D, h, w)
        context = self._build_context(z0, z1)
        context = self.context_norm(context)

        # 3. Target-time query map: (B, D, h, w)
        tau = tau.to(dtype=x.dtype, device=x.device)
        query = self.query_map(tau)

        # 4. Cross-attend query against context
        z = self.cross_attn(query, context)   # (B, D, h, w)

        # 5. Self-attention processor blocks
        for blk in self.blocks:
            if self.checkpointing_level >= 3:
                z = checkpoint(blk, z, use_reentrant=False)
            else:
                z = blk(z)

        # 6. Decode to output grid
        if self.checkpointing_level >= 1:
            out = checkpoint(self._decode, z, use_reentrant=False)
        else:
            out = self._decode(z)

        return out
