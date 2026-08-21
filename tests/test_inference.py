"""Tests for GroverModel (inference.py), using the real grover_base.pt
checkpoint loaded directly by path (never touches the network)."""

from pathlib import Path

import numpy as np
import pytest

from jax_grover.convert import load_grover_checkpoint
from jax_grover.embedding import ATOM_BRANCHES, BOND_BRANCHES
from jax_grover.inference import GroverModel
from jax_grover.weights.download_weights import weights_path

CHECKPOINT = Path(weights_path("base"))
FIXTURE = Path(__file__).parent / "fixtures" / "checkpoint_golden.npz"

pytestmark = pytest.mark.skipif(not CHECKPOINT.is_file(), reason="grover_base.pt not present locally")


@pytest.fixture(scope="module")
def preloaded_params_config():
    return load_grover_checkpoint(str(CHECKPOINT))


def _model(preloaded_params_config, **kwargs) -> GroverModel:
    params, config = preloaded_params_config
    return GroverModel(params=params, config=config, download=False, warmup=False, **kwargs)


def test_construction_requires_params_and_config_together(preloaded_params_config):
    params, config = preloaded_params_config
    with pytest.raises(ValueError):
        GroverModel(params=params, config=None, download=False, warmup=False)
    with pytest.raises(ValueError):
        GroverModel(params=None, config=config, download=False, warmup=False)


def test_embed_smiles_single_string_returns_1d(preloaded_params_config):
    model = _model(preloaded_params_config)
    fp = model.embed_smiles("c1ccccc1")
    assert fp.ndim == 1
    assert fp.shape[0] == model.config.hidden_size * 4  # "both" -> 4 branches
    assert np.isfinite(fp).all()


def test_embed_smiles_list_returns_2d(preloaded_params_config):
    model = _model(preloaded_params_config)
    fps = model.embed_smiles(["CCO", "c1ccccc1", "[NH4+]"])
    assert fps.shape == (3, model.config.hidden_size * 4)
    assert np.isfinite(fps).all()


def test_embed_smiles_empty_list(preloaded_params_config):
    model = _model(preloaded_params_config)
    fps = model.embed_smiles([])
    assert fps.shape == (0, 0)


@pytest.mark.parametrize("fingerprint_source,expected_factor", [("atom", 2), ("bond", 2), ("both", 4)])
def test_fingerprint_source_controls_output_width(preloaded_params_config, fingerprint_source, expected_factor):
    model = _model(preloaded_params_config, fingerprint_source=fingerprint_source)
    fp = model.embed_smiles("CCO")
    assert fp.shape[0] == model.config.hidden_size * expected_factor


def test_per_call_fingerprint_source_overrides_default(preloaded_params_config):
    model = _model(preloaded_params_config, fingerprint_source="both")
    fp_atom = model.embed_smiles("CCO", fingerprint_source="atom")
    assert fp_atom.shape[0] == model.config.hidden_size * 2


def test_chunking_does_not_change_results(preloaded_params_config):
    model = _model(preloaded_params_config)
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "[NH4+]", "C1CC1"]
    whole = model.embed_smiles(smiles_list, chunk_size=len(smiles_list))
    chunked = model.embed_smiles(smiles_list, chunk_size=2)
    np.testing.assert_allclose(whole, chunked, atol=1e-5, rtol=1e-5)


def test_embed_smiles_matches_unpadded_fingerprint(preloaded_params_config):
    """embed_smiles pads each chunk internally (graph.pad_batch); check it
    matches grover_fingerprint run unpadded on the same molecules."""
    from jax_grover.fingerprint import grover_fingerprint
    from jax_grover.graph import smiles_list_to_batch

    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "[NH4+]", "C1CC1"]
    model = _model(preloaded_params_config)
    padded_result = model.embed_smiles(smiles_list, chunk_size=len(smiles_list))

    params, config = preloaded_params_config
    batch = smiles_list_to_batch(smiles_list)
    unpadded_result = np.asarray(grover_fingerprint(params, config, batch, fingerprint_source=model.fingerprint_source))

    np.testing.assert_allclose(padded_result, unpadded_result, atol=1e-5, rtol=1e-5)


def test_embed_smiles_matches_golden_fixture(preloaded_params_config):
    """Cross-check GroverModel's wiring end-to-end against the same frozen
    golden fixture test_real_checkpoints.py verifies against."""
    fixture = np.load(FIXTURE)
    smiles_list = [str(s) for s in fixture["smiles"]]

    model = _model(preloaded_params_config)
    fps = model.embed_smiles(smiles_list)

    from jax_grover.graph import smiles_list_to_batch

    batch = smiles_list_to_batch(smiles_list)
    num_graphs = batch.num_graphs

    def naive_mean_pool(values, group_ids, num_groups):
        out = np.zeros((num_groups, values.shape[1]), dtype=values.dtype)
        for g in range(num_groups):
            rows = values[group_ids == g]
            if len(rows) > 0:
                out[g] = rows.mean(axis=0)
        return out

    expected_parts = []
    for key in ATOM_BRANCHES + BOND_BRANCHES:
        golden = fixture[f"base_{key}"]
        group_ids = batch.atom_batch if key.startswith("atom") else batch.edge_batch
        expected_parts.append(naive_mean_pool(golden, group_ids, num_graphs))
    expected = np.concatenate(expected_parts, axis=1)

    np.testing.assert_allclose(fps, expected, atol=1e-3, rtol=1e-3)
