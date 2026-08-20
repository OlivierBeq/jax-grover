"""Mean-pool encoder output into fixed-size per-molecule vectors
(``Readout(rtype="mean")``, the only readout this package's scope uses)."""

import functools

import jax
import jax.numpy as jnp

from .config import GroverConfig
from .encoder import grover_embedding_apply
from .graph import Batch
from .layers import readout_mean_apply

ATOM_BRANCHES: tuple[str, str] = ("atom_from_atom", "atom_from_bond")
BOND_BRANCHES: tuple[str, str] = ("bond_from_atom", "bond_from_bond")


@functools.partial(jax.jit, static_argnames=("cfg", "num_graphs"))
def _pool_grover_embeddings_jit(
    params: dict,
    cfg: GroverConfig,
    x: jnp.ndarray,
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray,
    rev_index: jnp.ndarray,
    atom_batch: jnp.ndarray,
    edge_batch: jnp.ndarray,
    num_graphs: int,
) -> dict[str, jnp.ndarray | None]:
    output = grover_embedding_apply(params, cfg, x, edge_index, edge_attr, rev_index)
    result: dict[str, jnp.ndarray | None] = {}
    for key in ATOM_BRANCHES:
        result[key] = readout_mean_apply(output[key], atom_batch, num_graphs) if output[key] is not None else None
    for key in BOND_BRANCHES:
        result[key] = readout_mean_apply(output[key], edge_batch, num_graphs) if output[key] is not None else None
    return result


def pool_grover_embeddings(params: dict, cfg: GroverConfig, batch: Batch) -> dict[str, jnp.ndarray | None]:
    """Run the encoder on ``batch``, mean-pool each populated branch.
    Branches absent under ``cfg.embedding_output_type`` stay ``None``.
    JIT-compiled: cached per ``(cfg, num_atoms, num_edges, num_graphs)``."""
    x = jnp.asarray(batch.x)
    edge_index = jnp.asarray(batch.edge_index)
    edge_attr = jnp.asarray(batch.edge_attr)
    rev_index = jnp.asarray(batch.rev_index)
    atom_batch = jnp.asarray(batch.atom_batch)
    edge_batch = jnp.asarray(batch.edge_batch)

    return _pool_grover_embeddings_jit(params, cfg, x, edge_index, edge_attr, rev_index, atom_batch, edge_batch, batch.num_graphs)
