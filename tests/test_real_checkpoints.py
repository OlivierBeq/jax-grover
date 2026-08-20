"""Fidelity tests against the real released GROVER checkpoints
(src/jax_grover/weights/, gitignored; download via
``download_grover_weights`` or ``load_grover_encoder(..., download=True)``).

Checked against a frozen golden fixture (tests/fixtures/checkpoint_golden.npz):
per-atom/per-edge embeddings from an independent PyTorch reference, pinned
as a permanent regression guard.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_grover.convert import load_grover_checkpoint
from jax_grover.encoder import grover_embedding_apply
from jax_grover.fingerprint import load_grover_encoder
from jax_grover.graph import smiles_list_to_batch
from jax_grover.weights.download_weights import weights_path

CHECKPOINTS = {"base": Path(weights_path("base")), "large": Path(weights_path("large"))}
FIXTURE = Path(__file__).parent / "fixtures" / "checkpoint_golden.npz"
BRANCH_KEYS = ("atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond")

pytestmark = pytest.mark.parametrize("model_type", ["base", "large"])


def _skip_if_absent(model_type):
    path = CHECKPOINTS[model_type]
    if not path.is_file():
        pytest.skip(f"{path} not present locally - run download_grover_weights() or see README")
    return path


def test_checkpoint_config_matches(model_type):
    path = _skip_if_absent(model_type)
    _, cfg = load_grover_checkpoint(str(path))
    expected_hidden = 800 if model_type == "base" else 1200
    assert cfg.hidden_size == expected_hidden
    assert cfg.depth == 6
    assert cfg.num_attn_head == 4
    assert cfg.num_mt_block == 1
    assert cfg.activation == "PReLU"
    assert cfg.embedding_output_type == "both"


def test_real_checkpoint_matches_golden_fixture(model_type):
    path = _skip_if_absent(model_type)
    fixture = np.load(FIXTURE)
    smiles_list = [str(s) for s in fixture["smiles"]]

    params, cfg = load_grover_checkpoint(str(path))
    batch = smiles_list_to_batch(smiles_list)
    out = grover_embedding_apply(params, cfg, jnp.asarray(batch.x), jnp.asarray(batch.edge_index), jnp.asarray(batch.edge_attr), jnp.asarray(batch.rev_index))

    for key in BRANCH_KEYS:
        golden = fixture[f"{model_type}_{key}"]
        ours = np.asarray(out[key])
        max_abs_diff = np.abs(golden - ours).max()
        np.testing.assert_allclose(ours, golden, atol=1e-3, rtol=1e-3, err_msg=f"branch={key} model={model_type} max_abs_diff={max_abs_diff}")


def test_load_grover_encoder_finds_local_checkpoint_without_redownloading(model_type):
    """``download=True`` must not error on an already-present checkpoint."""
    _skip_if_absent(model_type)
    params, cfg = load_grover_encoder(model_type=model_type, download=True)
    expected_hidden = 800 if model_type == "base" else 1200
    assert cfg.hidden_size == expected_hidden
    assert "encoders" in params
