"""End-to-end tests for ``embed_smiles``, using the real grover_base.pt
checkpoint. Expected fingerprints are computed independently here (plain
numpy mean-pooling of the golden fixture), not via jax_grover's own pooling
code, cross-checking ``pool_grover_embeddings``/``grover_fingerprint``.
"""

from pathlib import Path

import numpy as np
import pytest

from jax_grover.embedding import ATOM_BRANCHES, BOND_BRANCHES
from jax_grover.fingerprint import embed_smiles
from jax_grover.graph import smiles_list_to_batch
from jax_grover.weights.download_weights import weights_path

CHECKPOINT = Path(weights_path("base"))
FIXTURE = Path(__file__).parent / "fixtures" / "checkpoint_golden.npz"


def _naive_mean_pool(values: np.ndarray, group_ids: np.ndarray, num_groups: int) -> np.ndarray:
    out = np.zeros((num_groups, values.shape[1]), dtype=values.dtype)
    for g in range(num_groups):
        rows = values[group_ids == g]
        if len(rows) > 0:
            out[g] = rows.mean(axis=0)
    return out


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="grover_base.pt not present locally")
@pytest.mark.parametrize("fingerprint_source", ["atom", "bond", "both"])
def test_embed_smiles_matches_naive_pooling_of_golden_fixture(fingerprint_source):
    fixture = np.load(FIXTURE)
    smiles_list = [str(s) for s in fixture["smiles"]]
    batch = smiles_list_to_batch(smiles_list)
    num_graphs = batch.num_graphs

    branches = {
        "atom": ATOM_BRANCHES,
        "bond": BOND_BRANCHES,
        "both": ATOM_BRANCHES + BOND_BRANCHES,
    }[fingerprint_source]

    expected_parts = []
    for key in branches:
        golden = fixture[f"base_{key}"]
        group_ids = batch.atom_batch if key.startswith("atom") else batch.edge_batch
        expected_parts.append(_naive_mean_pool(golden, group_ids, num_graphs))
    expected = np.concatenate(expected_parts, axis=1)

    ours = np.asarray(embed_smiles(smiles_list, checkpoint_path=str(CHECKPOINT), fingerprint_source=fingerprint_source))

    np.testing.assert_allclose(ours, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="grover_base.pt not present locally")
def test_embed_smiles_single_molecule():
    fp = embed_smiles(["c1ccccc1"], checkpoint_path=str(CHECKPOINT))
    assert fp.shape[0] == 1
    assert np.isfinite(np.asarray(fp)).all()
