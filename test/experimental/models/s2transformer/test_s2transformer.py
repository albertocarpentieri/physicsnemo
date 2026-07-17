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
"""Tests for the experimental S2Transformer (:class:`S2Transformer`).

All tests build a tiny CPU-friendly model (global attention, encoder/decoder
auto-disabled) and check:

* a forward pass produces the expected shape;
* a physicsnemo ``Module`` save/reload round-trip reproduces the output (this
  also exercises the nested-config -> ``asdict`` -> reconstruct serialization);
* FiLM stochastic conditioning (``noise.mode="film"``): zero-init warm-start
  exactness, latent-driven diversity, gradient flow, and rejection of unknown
  modes.
"""

from __future__ import annotations

import pytest
import torch

# The model requires torch_harmonics; skip the whole module if it is missing.
pytest.importorskip("torch_harmonics")

from physicsnemo.experimental.models.s2transformer import (
    NoiseConfig,
    S2PosEmbedConfig,
    S2ProcessorConfig,
    S2Transformer,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _tiny_model(**overrides) -> S2Transformer:
    """A small CPU-friendly model: global attention, no encoder/decoder.

    With ``in_channels == out_channels == embed_dim`` and ``scale_factor=1`` the
    grid/channels don't change between input, latent, and output, so the
    encoder/decoder auto-deactivate (nn.Identity) and the processor runs directly
    on the input. ``attention_mode="global"`` + ``attn_optimized_kernel=False``
    avoids the neighborhood-attention CUDA kernel, so the model runs on CPU.
    """
    kwargs = dict(
        model_grid_type="equiangular",
        sht_grid_type="equiangular",
        inp_shape=(16, 32),
        out_shape=(16, 32),
        in_channels=8,
        out_channels=8,
        scale_factor=1,
        processor=S2ProcessorConfig(
            embed_dim=8,
            num_layers=2,
            num_heads=4,
            attention_mode="global",
            attn_optimized_kernel=False,
            normalization="layer_norm",
        ),
        pos_embed=S2PosEmbedConfig(kind="spectral"),
    )
    kwargs.update(overrides)
    return S2Transformer(**kwargs)


# -----------------------------------------------------------------------------
# Forward + checkpoint round-trip
# -----------------------------------------------------------------------------


def test_forward_shape():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    x = torch.randn(2, 8, 16, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 8, 16, 32)
    assert torch.isfinite(y).all()


def test_encoder_decoder_auto_activation():
    # equal grid + channels -> encoder/decoder collapse to Identity
    m = _tiny_model()
    assert isinstance(m.encoder, torch.nn.Identity)
    assert isinstance(m.decoder, torch.nn.Identity)
    assert m.use_encoder is False and m.use_decoder is False
    # a channel change auto-activates both (construction only; no CPU forward)
    m2 = _tiny_model(in_channels=4, out_channels=1)
    assert m2.use_encoder and m2.use_decoder


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model().eval()
    x = torch.randn(1, 8, 16, 32)
    with torch.no_grad():
        y_ref = model(x)

    ckpt = tmp_path / "s2transformer.mdlus"
    model.save(str(ckpt))
    reloaded = S2Transformer.from_checkpoint(str(ckpt)).eval()
    with torch.no_grad():
        y_new = reloaded(x)

    assert torch.allclose(y_ref, y_new, atol=1e-6, rtol=0)


# -----------------------------------------------------------------------------
# FiLM stochastic conditioning (noise.mode="film")
# -----------------------------------------------------------------------------


def test_film_zero_init_is_identity_and_diversifies():
    """Zero-init FiLM heads reproduce the deterministic map exactly at init
    (warm-start safety); once the heads are non-zero, different latents give
    different outputs."""
    torch.manual_seed(0)
    model = _tiny_model(noise=NoiseConfig(mode="film", latent_dim=8)).eval()
    x = torch.randn(2, 8, 16, 32)
    z = torch.randn(2, 8)
    with torch.no_grad():
        y_none = model(x)                        # deterministic path (no latent)
        y_zero = model(x, film_latent=z)         # zero-init heads -> identity
    assert torch.allclose(y_none, y_zero, atol=1e-6), (
        "zero-init FiLM must reproduce the deterministic output at init"
    )

    # Activate the FiLM heads, then check the latent actually modulates output.
    with torch.no_grad():
        for blk in model.blocks:
            if blk.film0 is not None:
                for head in (blk.film0, blk.film1):
                    head.weight.normal_(0.0, 0.1)
                    head.bias.normal_(0.0, 0.1)
        y_a = model(x, film_latent=z)
        y_b = model(x, film_latent=torch.randn(2, 8))
    assert not torch.allclose(y_a, y_none, atol=1e-5), "active FiLM must change the output"
    assert not torch.allclose(y_a, y_b, atol=1e-5), "different latents must differ"
    assert torch.isfinite(y_a).all()


def test_film_gradients_flow():
    """Gradients reach the FiLM latent and the conditioning MLP."""
    torch.manual_seed(0)
    model = _tiny_model(noise=NoiseConfig(mode="film", latent_dim=8))
    for blk in model.blocks:                     # break the zero-init identity
        if blk.film0 is not None:
            torch.nn.init.normal_(blk.film0.weight, 0.0, 0.1)
            torch.nn.init.normal_(blk.film1.weight, 0.0, 0.1)
    x = torch.randn(1, 8, 16, 32)
    z = torch.randn(1, 8, requires_grad=True)
    model(x, film_latent=z).pow(2).mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    film_grads = [p.grad for p in model.film_embed.parameters() if p.grad is not None]
    assert film_grads and any(g.abs().sum() > 0 for g in film_grads)


def test_film_rejects_unknown_noise_mode():
    with pytest.raises(NotImplementedError):
        _tiny_model(noise=NoiseConfig(mode="concatenate"))
