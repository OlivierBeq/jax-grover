"""Graph-construction + aggregation tests, checked against naive independent
Python-loop reimplementations (not the vectorized production code path)."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_grover.batch_ops import aggregate_atom_view, aggregate_bond_view, segment_mean
from jax_grover.graph import pad_batch, smiles_list_to_batch, smiles_to_data

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


def test_pad_batch_leaves_real_atoms_edges_graphs_untouched():
    batch = smiles_list_to_batch(TEST_SMILES)
    padded = pad_batch(batch, num_graphs=batch.num_graphs + 1)

    real_atoms = batch.x.shape[0]
    real_edges = batch.edge_index.shape[1]

    np.testing.assert_array_equal(padded.x[:real_atoms], batch.x)
    np.testing.assert_array_equal(padded.edge_index[:, :real_edges], batch.edge_index)
    np.testing.assert_array_equal(padded.edge_attr[:real_edges], batch.edge_attr)
    np.testing.assert_array_equal(padded.rev_index[:real_edges], batch.rev_index)
    np.testing.assert_array_equal(padded.atom_batch[:real_atoms], batch.atom_batch)
    np.testing.assert_array_equal(padded.edge_batch[:real_edges], batch.edge_batch)

    assert (padded.edge_index[:, real_edges:] >= real_atoms).all()  # never a real atom
    assert (padded.atom_batch[real_atoms:] == padded.num_graphs - 1).all()  # dedicated pad slot
    if padded.edge_batch.shape[0] > real_edges:
        assert (padded.edge_batch[real_edges:] == padded.num_graphs - 1).all()


def test_pad_batch_preserves_aggregation_for_real_atoms():
    batch = smiles_list_to_batch(TEST_SMILES)
    padded = pad_batch(batch, num_graphs=batch.num_graphs + 1)
    real_atoms = batch.x.shape[0]
    n_pad_atoms = padded.x.shape[0] - real_atoms

    rng = np.random.default_rng(0)
    message = rng.standard_normal((real_atoms, 8)).astype(np.float32)
    padded_message = np.concatenate([message, rng.standard_normal((n_pad_atoms, 8)).astype(np.float32)])

    unpadded_out = np.asarray(aggregate_atom_view(jnp.asarray(message), jnp.asarray(batch.edge_index), real_atoms))
    padded_out = np.asarray(aggregate_atom_view(jnp.asarray(padded_message), jnp.asarray(padded.edge_index), padded.x.shape[0]))

    np.testing.assert_array_equal(padded_out[:real_atoms], unpadded_out)  # non-zero pad message can't leak in


def test_pad_batch_rejects_insufficient_num_graphs():
    batch = smiles_list_to_batch(TEST_SMILES)
    with pytest.raises(ValueError):
        pad_batch(batch, num_graphs=batch.num_graphs)


def test_pad_batch_shapes_are_bucketed_and_stable_across_sizes():
    small = pad_batch(smiles_list_to_batch(TEST_SMILES[:2]), num_graphs=257)
    large = pad_batch(smiles_list_to_batch(TEST_SMILES), num_graphs=257)

    # Both tiny enough to land in the same bucket -> same compiled shape.
    assert small.x.shape[0] == large.x.shape[0]
    assert small.edge_index.shape[1] == large.edge_index.shape[1]
    assert small.num_graphs == large.num_graphs == 257
