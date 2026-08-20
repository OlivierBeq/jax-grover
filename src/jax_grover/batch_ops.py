"""Scatter-sum neighbor aggregation for the dual-view encoder (COO
``edge_index``/``rev_index`` representation, via ``jax.ops.segment_sum``).
"""

import jax.numpy as jnp
from jax import ops as jax_ops


def segment_sum(data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int) -> jnp.ndarray:
    return jax_ops.segment_sum(data, segment_ids, num_segments=num_segments)


def aggregate_atom_view(message: jnp.ndarray, edge_index: jnp.ndarray, num_atoms: int) -> jnp.ndarray:
    """Atom-view neighbor aggregation: sum incoming per-atom messages."""
    src, dst = edge_index[0], edge_index[1]
    return segment_sum(message[src], dst, num_atoms)


def aggregate_edge_attr_to_atom(edge_attr: jnp.ndarray, edge_index: jnp.ndarray, num_atoms: int) -> jnp.ndarray:
    """Sum per-edge features into their target atom."""
    return segment_sum(edge_attr, edge_index[1], num_atoms)


def aggregate_bond_view(message: jnp.ndarray, edge_index: jnp.ndarray, rev_index: jnp.ndarray, num_atoms: int) -> jnp.ndarray:
    """Bond-view (D-MPNN-style) aggregation: sum incoming messages at an
    edge's source atom, minus that edge's own reverse contribution."""
    src, dst = edge_index[0], edge_index[1]
    incoming_sum_per_atom = segment_sum(message, dst, num_atoms)
    a_message = incoming_sum_per_atom[src]
    return a_message - message[rev_index]


def aggregate_atom_to_bond(atom_message: jnp.ndarray, edge_index: jnp.ndarray, num_atoms: int) -> jnp.ndarray:
    """Per-edge (src -> dst): ``atom_message[src] + sum(neighbors(src)) - atom_message[dst]``."""
    src, dst = edge_index[0], edge_index[1]
    neighbor_sum = aggregate_atom_view(atom_message, edge_index, num_atoms)
    return atom_message[src] + neighbor_sum[src] - atom_message[dst]


def segment_mean(data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int) -> jnp.ndarray:
    """Mean-pool rows by ``segment_ids``; an empty group yields zeros (not NaN)."""
    sums = segment_sum(data, segment_ids, num_segments)
    counts = segment_sum(jnp.ones((data.shape[0], 1), dtype=data.dtype), segment_ids, num_segments)
    return jnp.where(counts > 0, sums / jnp.where(counts > 0, counts, 1.0), 0.0)
