"""Encoder architectural-invariant tests on randomly-initialized params:
determinism, correct output shapes, and (most importantly)
batch-composition independence - a molecule's embedding must not depend on
which other molecules share its batch.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_grover.config import GroverConfig
from jax_grover.convert import edge_node_fdim
from jax_grover.encoder import grover_embedding_apply
from jax_grover.graph import smiles_list_to_batch
from jax_grover.init import init_grover_embedding_params

from .conftest import TEST_SMILES

BRANCH_KEYS = ("atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond")


def _run(params, cfg, smiles_list):
    batch = smiles_list_to_batch(smiles_list)
    return grover_embedding_apply(
        params, cfg, jnp.asarray(batch.x), jnp.asarray(batch.edge_index), jnp.asarray(batch.edge_attr), jnp.asarray(batch.rev_index)
    ), batch


@pytest.mark.parametrize(
    "bias,dense,activation,embedding_output_type",
    list(itertools.product([False, True], [False, True], ["ReLU", "PReLU"], ["atom", "bond", "both"])),
)
def test_output_shapes_and_finiteness(bias, dense, activation, embedding_output_type):
    hidden_size = 16
    cfg = GroverConfig(
        hidden_size=hidden_size, bias=bias, depth=3, dropout=0.0, activation=activation, dense=dense,
        num_mt_block=1, num_attn_head=2, embedding_output_type=embedding_output_type,
    )
    edge_fdim, node_fdim = edge_node_fdim()
    params = init_grover_embedding_params(jax.random.PRNGKey(0), cfg, edge_fdim, node_fdim)

    out, batch = _run(params, cfg, TEST_SMILES)
    num_atoms = batch.x.shape[0]
    num_edges = batch.edge_index.shape[1]

    for key in ("atom_from_atom", "atom_from_bond"):
        if embedding_output_type in ("atom", "both"):
            assert out[key].shape == (num_atoms, hidden_size)
            assert bool(jnp.isfinite(out[key]).all())
        else:
            assert out[key] is None
    for key in ("bond_from_atom", "bond_from_bond"):
        if embedding_output_type in ("bond", "both"):
            assert out[key].shape == (num_edges, hidden_size)
            assert bool(jnp.isfinite(out[key]).all())
        else:
            assert out[key] is None


def test_deterministic_across_repeated_calls():
    cfg = GroverConfig(hidden_size=16, depth=3, num_mt_block=1, num_attn_head=2, activation="PReLU")
    edge_fdim, node_fdim = edge_node_fdim()
    params = init_grover_embedding_params(jax.random.PRNGKey(1), cfg, edge_fdim, node_fdim)

    out1, _ = _run(params, cfg, TEST_SMILES)
    out2, _ = _run(params, cfg, TEST_SMILES)
    for key in BRANCH_KEYS:
        np.testing.assert_array_equal(np.asarray(out1[key]), np.asarray(out2[key]))


def test_multi_block_multi_head_runs_and_is_finite():
    cfg = GroverConfig(hidden_size=24, depth=4, num_mt_block=2, num_attn_head=3, activation="PReLU")
    edge_fdim, node_fdim = edge_node_fdim()
    params = init_grover_embedding_params(jax.random.PRNGKey(2), cfg, edge_fdim, node_fdim)
    out, _ = _run(params, cfg, TEST_SMILES)
    for key in BRANCH_KEYS:
        assert bool(jnp.isfinite(out[key]).all())


@pytest.mark.parametrize("activation", ["ReLU", "PReLU"])
def test_batch_composition_independence(activation):
    """A molecule's embedding must be identical whether encoded alone or
    batched with others, regardless of batch order."""
    cfg = GroverConfig(hidden_size=16, depth=3, num_mt_block=1, num_attn_head=2, activation=activation)
    edge_fdim, node_fdim = edge_node_fdim()
    params = init_grover_embedding_params(jax.random.PRNGKey(3), cfg, edge_fdim, node_fdim)

    target = "c1ccc2ccccc2c1"  # naphthalene - fused rings, heterogeneous atom degree
    solo_out, solo_batch = _run(params, cfg, [target])

    for smiles_list in ([target, "CCO", "CC(=O)O"], ["CCO", target, "[NH4+]"], ["CC(N)C(=O)O", "C1CCCCCCC1", target]):
        idx = smiles_list.index(target)
        batched_out, batched_batch = _run(params, cfg, smiles_list)

        atom_mask = np.asarray(batched_batch.atom_batch) == idx
        edge_mask = np.asarray(batched_batch.edge_batch) == idx

        np.testing.assert_allclose(np.asarray(batched_out["atom_from_atom"])[atom_mask], np.asarray(solo_out["atom_from_atom"]), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(batched_out["atom_from_bond"])[atom_mask], np.asarray(solo_out["atom_from_bond"]), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(batched_out["bond_from_atom"])[edge_mask], np.asarray(solo_out["bond_from_atom"]), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(batched_out["bond_from_bond"])[edge_mask], np.asarray(solo_out["bond_from_bond"]), atol=1e-5, rtol=1e-5)


def test_single_molecule_and_zero_bond_molecule_do_not_crash():
    cfg = GroverConfig(hidden_size=16, depth=3, num_mt_block=1, num_attn_head=2, activation="ReLU")
    edge_fdim, node_fdim = edge_node_fdim()
    params = init_grover_embedding_params(jax.random.PRNGKey(4), cfg, edge_fdim, node_fdim)

    out, batch = _run(params, cfg, ["[NH4+]", "CCO", "c1ccccc1"])
    for key in BRANCH_KEYS:
        assert bool(jnp.isfinite(out[key]).all())


def test_batch_with_zero_total_bonds_does_not_crash():
    """Every molecule is a single bond-free atom -> edge_index has 0 columns.
    Regression test: used to crash on an ambiguous ``reshape(n, -1, ...)``
    with n == 0 (see layers.multi_headed_attention_apply)."""
    cfg = GroverConfig(hidden_size=16, depth=3, num_mt_block=1, num_attn_head=2, activation="PReLU")
    edge_fdim, node_fdim = edge_node_fdim()
    params = init_grover_embedding_params(jax.random.PRNGKey(5), cfg, edge_fdim, node_fdim)

    out, batch = _run(params, cfg, ["[NH4+]", "[Na+]", "[Cl-]"])
    assert batch.edge_index.shape[1] == 0
    for key in BRANCH_KEYS:
        expected_rows = batch.x.shape[0] if key.startswith("atom") else batch.edge_index.shape[1]
        assert out[key].shape[0] == expected_rows
        assert bool(jnp.isfinite(out[key]).all())
