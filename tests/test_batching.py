"""Graph-construction + aggregation tests, checked against naive independent
Python-loop reimplementations (not the vectorized production code path)."""

import jax.numpy as jnp
import numpy as np

from jax_grover.batch_ops import aggregate_atom_view, aggregate_bond_view, segment_mean
from jax_grover.graph import smiles_list_to_batch, smiles_to_data

from .conftest import TEST_SMILES


def _naive_aggregate_atom_view(message, edge_index, num_atoms):
    out = np.zeros((num_atoms, message.shape[1]), dtype=message.dtype)
    for e in range(edge_index.shape[1]):
        src, dst = edge_index[0, e], edge_index[1, e]
        out[dst] += message[src]
    return out


def _naive_aggregate_bond_view(message, edge_index, rev_index, num_atoms):
    incoming = np.zeros((num_atoms, message.shape[1]), dtype=message.dtype)
    for e in range(edge_index.shape[1]):
        dst = edge_index[1, e]
        incoming[dst] += message[e]
    out = np.zeros_like(message)
    for e in range(edge_index.shape[1]):
        src = edge_index[0, e]
        out[e] = incoming[src] - message[rev_index[e]]
    return out


def test_batch_construction_shapes_and_edge_pairing():
    batch = smiles_list_to_batch(TEST_SMILES)
    assert batch.x.shape[1] == 151
    assert batch.edge_attr.shape[1] == 14
    assert batch.edge_index.shape[1] % 2 == 0  # bonds always added as reverse pairs
    assert batch.num_graphs == len(TEST_SMILES)
    assert batch.atom_batch.shape[0] == batch.x.shape[0]
    assert batch.atom_batch.max() == len(TEST_SMILES) - 1

    # Directed edges 2k/2k+1 are always mutual reverses (rev_index[2k] == 2k+1).
    n_edges = batch.edge_index.shape[1]
    for k in range(0, n_edges, 2):
        assert batch.rev_index[k] == k + 1
        assert batch.rev_index[k + 1] == k
        assert (batch.edge_index[:, k] == batch.edge_index[::-1, k + 1]).all()


def test_zero_bond_molecule_has_no_edges():
    data = smiles_to_data("[NH4+]")  # single atom, zero bonds
    assert data.x.shape[0] == 1
    assert data.edge_index.shape == (2, 0)


def test_batching_offsets_indices_correctly():
    batch = smiles_list_to_batch(["CCO", "c1ccccc1"])
    single_ethanol = smiles_to_data("CCO")
    single_benzene = smiles_to_data("c1ccccc1")

    n_atoms_ethanol = single_ethanol.x.shape[0]
    n_edges_ethanol = single_ethanol.edge_index.shape[1]

    np.testing.assert_array_equal(batch.x[:n_atoms_ethanol], single_ethanol.x)
    np.testing.assert_array_equal(batch.x[n_atoms_ethanol:], single_benzene.x)
    np.testing.assert_array_equal(batch.edge_index[:, :n_edges_ethanol], single_ethanol.edge_index)
    # benzene's edges must be offset by ethanol's atom count
    np.testing.assert_array_equal(batch.edge_index[:, n_edges_ethanol:], single_benzene.edge_index + n_atoms_ethanol)


def test_aggregate_atom_view_matches_naive_reference():
    batch = smiles_list_to_batch(TEST_SMILES)
    num_atoms = batch.x.shape[0]
    rng = np.random.default_rng(0)
    message = rng.standard_normal((num_atoms, 8)).astype(np.float32)

    ours = np.asarray(aggregate_atom_view(jnp.asarray(message), jnp.asarray(batch.edge_index), num_atoms))
    naive = _naive_aggregate_atom_view(message, batch.edge_index, num_atoms)

    np.testing.assert_allclose(ours, naive, atol=1e-6)


def test_aggregate_bond_view_matches_naive_reference():
    batch = smiles_list_to_batch(TEST_SMILES)
    num_atoms = batch.x.shape[0]
    num_edges = batch.edge_index.shape[1]
    rng = np.random.default_rng(0)
    message = rng.standard_normal((num_edges, 8)).astype(np.float32)

    ours = np.asarray(aggregate_bond_view(jnp.asarray(message), jnp.asarray(batch.edge_index), jnp.asarray(batch.rev_index), num_atoms))
    naive = _naive_aggregate_bond_view(message, batch.edge_index, batch.rev_index, num_atoms)

    np.testing.assert_allclose(ours, naive, atol=1e-6)


def test_segment_mean_matches_naive_reference_and_handles_empty_groups():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((7, 3)).astype(np.float32)
    # group 2 is intentionally empty (no rows with segment_id == 2)
    segment_ids = np.array([0, 0, 1, 0, 3, 3, 1], dtype=np.int32)
    num_segments = 4

    ours = np.asarray(segment_mean(jnp.asarray(data), jnp.asarray(segment_ids), num_segments))

    naive = np.zeros((num_segments, 3), dtype=np.float32)
    for g in range(num_segments):
        rows = data[segment_ids == g]
        if len(rows) > 0:
            naive[g] = rows.mean(axis=0)

    np.testing.assert_allclose(ours, naive, atol=1e-6)
    np.testing.assert_array_equal(ours[2], np.zeros(3, dtype=np.float32))  # empty group -> zero vector
