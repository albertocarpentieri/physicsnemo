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
"""Tests for the experimental S2Transformer (:class:`PrecipAttentionNet`).

Two always-on tests build a tiny CPU-friendly model (global attention, no
encoder/decoder) to check (1) a forward pass and (2) a physicsnemo ``Module``
save/reload round-trip.

A third test loads a *real trained* ``PrecipAttentionNet`` checkpoint into the
new experimental code and runs a forward pass. It is skipped unless the
``PRECIP_ATTENTION_CHECKPOINT`` environment variable points at a ``.mdlus``
file. The same loader is exposed as a CLI so it can be run directly against a
trained checkpoint inside the training container::

    python test/experimental/models/s2transformer/test_precip_attention_net.py \
        --checkpoint /path/to/PrecipAttentionNet.0.<epoch>.mdlus --compare-old

``--compare-old`` additionally loads the legacy
``physicsnemo.models.lno.attention_net.PrecipAttentionNet`` from the same
checkpoint and asserts the new experimental code reproduces its output exactly.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import zipfile
from typing import Any, Dict, Tuple

import pytest
import torch

# The model requires torch_harmonics; skip the whole module if it is missing.
pytest.importorskip("torch_harmonics")

from physicsnemo.experimental.models.s2transformer import PrecipAttentionNet


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _tiny_model(**overrides) -> PrecipAttentionNet:
    """A small CPU-friendly model: global attention, no encoder/decoder.

    Using ``attention_mode="global"`` with ``use_encoder=False`` /
    ``use_decoder=False`` avoids the neighborhood-attention CUDA kernel, so the
    model runs on CPU. Channel routing: block 0 maps ``in_channels -> embed_dim``
    and the last block maps ``embed_dim -> out_channels`` inside the MLP.
    """
    kwargs = dict(
        model_grid_type="equiangular",
        sht_grid_type="equiangular",
        inp_shape=(16, 32),
        out_shape=(16, 32),
        in_channels=4,
        out_channels=1,
        scale_factor=1,
        embed_dim=8,
        num_layers=2,
        num_heads=4,
        attention_mode="global",
        attn_optimized_kernel=False,
        use_encoder=False,
        use_decoder=False,
        pos_embed="spectral",
        normalization_layer="layer_norm",
    )
    kwargs.update(overrides)
    return PrecipAttentionNet(**kwargs)


def _read_mdlus(path: str) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Return ``(args, state_dict)`` from a physicsnemo ``.mdlus`` archive.

    A ``.mdlus`` is a tar or zip archive containing ``args.json`` (with the
    ``__name__`` / ``__module__`` / ``__args__`` used to build the model) and
    ``model.pt`` (the state dict). Reading them directly lets us rebuild the
    model with the *new* class regardless of the ``__module__`` baked into the
    checkpoint at training time.
    """
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r") as tar:
            args = json.loads(tar.extractfile("args.json").read().decode("utf-8"))
            state_dict = torch.load(
                io.BytesIO(tar.extractfile("model.pt").read()),
                map_location="cpu",
                weights_only=False,
            )
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as archive:
            args = json.loads(archive.read("args.json").decode("utf-8"))
            state_dict = torch.load(
                io.BytesIO(archive.read("model.pt")),
                map_location="cpu",
                weights_only=False,
            )
    else:
        raise ValueError(
            f"{path} is not a .mdlus archive (expected a tar/zip with model.pt)."
        )
    return args, state_dict


def _grid_from_args(ctor: Dict[str, Any]) -> Tuple[int, Tuple[int, int], Tuple[int, int], int]:
    """Extract ``(in_channels, inp_shape, out_shape, out_channels)`` from ctor args."""
    inp = tuple(ctor.get("inp_shape", (721, 1440)))
    out = tuple(ctor.get("out_shape", inp))
    in_ch = int(ctor.get("in_channels", 27))
    out_ch = int(ctor.get("out_channels", 1))
    return in_ch, inp, out, out_ch


def load_trained_checkpoint(
    path: str, device: str = "cpu"
) -> Tuple[PrecipAttentionNet, Dict[str, Any]]:
    """Build the new :class:`PrecipAttentionNet` from a trained ``.mdlus`` and load its weights.

    Parameters
    ----------
    path : str
        Path to a ``PrecipAttentionNet`` ``.mdlus`` checkpoint.
    device : str, optional, default="cpu"
        Device to move the loaded model to.

    Returns
    -------
    tuple
        ``(model, ctor_args)`` where ``model`` is the loaded, ``eval()``-mode
        model and ``ctor_args`` is the constructor-argument dict from the
        checkpoint.
    """
    args, state_dict = _read_mdlus(path)
    ctor = dict(args.get("__args__", {}))
    model = PrecipAttentionNet(**ctor)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"state_dict mismatch loading {path} into the new PrecipAttentionNet:\n"
            f"  missing={list(incompatible.missing_keys)}\n"
            f"  unexpected={list(incompatible.unexpected_keys)}"
        )
    return model.to(device).eval(), ctor


# -----------------------------------------------------------------------------
# Always-on tests (tiny model, CPU)
# -----------------------------------------------------------------------------


def test_forward_shape():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    x = torch.randn(2, 4, 16, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 16, 32)
    assert torch.isfinite(y).all()


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model().eval()
    x = torch.randn(1, 4, 16, 32)
    with torch.no_grad():
        y_ref = model(x)

    ckpt = tmp_path / "precip_attention.mdlus"
    model.save(str(ckpt))
    reloaded = PrecipAttentionNet.from_checkpoint(str(ckpt)).eval()
    with torch.no_grad():
        y_new = reloaded(x)

    assert torch.allclose(y_ref, y_new, atol=1e-6, rtol=0)


# -----------------------------------------------------------------------------
# Trained-checkpoint test (real weights) - opt-in via env var
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("PRECIP_ATTENTION_CHECKPOINT"),
    reason="set PRECIP_ATTENTION_CHECKPOINT=<path to .mdlus> to test a trained checkpoint",
)
def test_load_trained_checkpoint():
    path = os.environ["PRECIP_ATTENTION_CHECKPOINT"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ctor = load_trained_checkpoint(path, device)
    in_ch, inp, out, out_ch = _grid_from_args(ctor)
    x = torch.randn(1, in_ch, *inp, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, out_ch, *out)
    assert torch.isfinite(y).all()


# -----------------------------------------------------------------------------
# CLI: run directly against a trained checkpoint
# -----------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained PrecipAttentionNet .mdlus with the new experimental "
            "physicsnemo code and run a forward pass."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a PrecipAttentionNet .mdlus (e.g. core.mdlus or "
        "PrecipAttentionNet.0.<epoch>.mdlus).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--compare-old",
        action="store_true",
        help="Also load physicsnemo.models.lno.attention_net.PrecipAttentionNet "
        "from the same checkpoint and assert identical output.",
    )
    args = parser.parse_args()

    model, ctor = load_trained_checkpoint(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    in_ch, inp, out, out_ch = _grid_from_args(ctor)
    print(f"[s2transformer] loaded {args.checkpoint}")
    print(
        "[s2transformer] class="
        "physicsnemo.experimental.models.s2transformer.PrecipAttentionNet "
        f"params={n_params:,}"
    )
    print(f"[s2transformer] in={in_ch}x{inp}  out={out_ch}x{out}  device={args.device}")

    torch.manual_seed(0)
    x = torch.randn(args.batch, in_ch, *inp, device=args.device)
    with torch.no_grad():
        y = model(x)
    print(
        f"[s2transformer] output shape={tuple(y.shape)} min={y.min():.4f} "
        f"max={y.max():.4f} mean={y.mean():.4f} finite={bool(torch.isfinite(y).all())}"
    )
    assert y.shape == (args.batch, out_ch, *out), y.shape
    assert torch.isfinite(y).all()

    if args.compare_old:
        from physicsnemo.models.lno.attention_net import (
            PrecipAttentionNet as LegacyPrecipAttentionNet,
        )

        legacy_args, state_dict = _read_mdlus(args.checkpoint)
        legacy = LegacyPrecipAttentionNet(**dict(legacy_args.get("__args__", {})))
        legacy = legacy.to(args.device).eval()
        legacy.load_state_dict(state_dict, strict=True)
        with torch.no_grad():
            y_legacy = legacy(x)
        max_abs = (y - y_legacy).abs().max().item()
        print(f"[s2transformer] max|new-legacy| = {max_abs:.3e}")
        assert torch.allclose(y, y_legacy, atol=1e-5, rtol=0), (
            f"new experimental output differs from legacy (max abs diff {max_abs:.3e})"
        )
        print("[s2transformer] OK: new experimental code matches legacy output.")

    print("[s2transformer] SMOKE TEST PASSED")


if __name__ == "__main__":
    _main()
