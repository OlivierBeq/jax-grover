"""The dual-view (atom-view + bond-view) GTransformer encoder.

Functional JAX implementation of GROVER's ``GTransEncoder``/``GROVEREmbedding``.
"""


import jax.numpy as jnp

from .batch_ops import aggregate_atom_to_bond, aggregate_atom_view, aggregate_bond_view, aggregate_edge_attr_to_atom
from .config import GroverConfig
from .layers import mt_block_apply, positionwise_feedforward_apply, sublayer_connection_apply


def bond_init_message(x: jnp.ndarray, edge_index: jnp.ndarray, edge_attr: jnp.ndarray) -> jnp.ndarray:
    """Directed-bond init message: concat(source atom features, bond features)."""
    return jnp.concatenate([x[edge_index[0]], edge_attr], axis=-1)


def _atom_bond_transform(
    params: dict,
    cfg: GroverConfig,
    to_atom: bool,
    atom_output: jnp.ndarray,
    bond_output: jnp.ndarray,
    original_f_atoms: jnp.ndarray,
    original_f_bonds: jnp.ndarray,
    edge_index: jnp.ndarray,
    rev_index: jnp.ndarray,
    num_atoms: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if to_atom:
        aggr_atom = aggregate_atom_view(atom_output, edge_index, num_atoms)
        atom_in_atom_out = sublayer_connection_apply(
            params["atom_from_atom_sublayer"],
            None,
            positionwise_feedforward_apply(params["ffn_atom_from_atom"], cfg, jnp.concatenate([original_f_atoms, aggr_atom], axis=1)),
        )

        aggr_bond = aggregate_edge_attr_to_atom(bond_output, edge_index, num_atoms)
        bond_in_atom_out = sublayer_connection_apply(
            params["atom_from_bond_sublayer"],
            None,
            positionwise_feedforward_apply(params["ffn_atom_from_bond"], cfg, jnp.concatenate([original_f_atoms, aggr_bond], axis=1)),
        )
        return atom_in_atom_out, bond_in_atom_out
    else:
        aggr_atom_to_bond = aggregate_atom_to_bond(atom_output, edge_index, num_atoms)
        atom_in_bond_out = sublayer_connection_apply(
            params["bond_from_atom_sublayer"],
            None,
            positionwise_feedforward_apply(params["ffn_bond_from_atom"], cfg, jnp.concatenate([original_f_bonds, aggr_atom_to_bond], axis=1)),
        )

        aggr_bond_to_bond = aggregate_bond_view(bond_output, edge_index, rev_index, num_atoms)
        bond_in_bond_out = sublayer_connection_apply(
            params["bond_from_bond_sublayer"],
            None,
            positionwise_feedforward_apply(params["ffn_bond_from_bond"], cfg, jnp.concatenate([original_f_bonds, aggr_bond_to_bond], axis=1)),
        )
        return atom_in_bond_out, bond_in_bond_out


def gtrans_encoder_apply(
    params: dict,
    cfg: GroverConfig,
    num_attn_head: int,
    atom_emb_output: str | None,
    x: jnp.ndarray,
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray,
    rev_index: jnp.ndarray,
):
    num_atoms = x.shape[0]

    original_f_atoms = x
    original_f_bonds = bond_init_message(x, edge_index, edge_attr)

    node_hidden = original_f_atoms
    for block_params in params["node_blocks"]:
        node_hidden = mt_block_apply(block_params, cfg, num_attn_head, True, node_hidden, edge_index, rev_index, num_atoms)

    edge_hidden = original_f_bonds
    for block_params in params["edge_blocks"]:
        edge_hidden = mt_block_apply(block_params, cfg, num_attn_head, False, edge_hidden, edge_index, rev_index, num_atoms)

    atom_output, bond_output = node_hidden, edge_hidden

    if atom_emb_output is None:
        return atom_output, bond_output

    rest = (atom_output, bond_output, original_f_atoms, original_f_bonds, edge_index, rev_index, num_atoms)
    if atom_emb_output == "atom":
        return _atom_bond_transform(params, cfg, True, *rest)
    elif atom_emb_output == "bond":
        return _atom_bond_transform(params, cfg, False, *rest)
    else:  # "both"
        atom_embeddings = _atom_bond_transform(params, cfg, True, *rest)
        bond_embeddings = _atom_bond_transform(params, cfg, False, *rest)
        return (atom_embeddings[0], bond_embeddings[0]), (atom_embeddings[1], bond_embeddings[1])


def grover_embedding_apply(
    params: dict, cfg: GroverConfig, x: jnp.ndarray, edge_index: jnp.ndarray, edge_attr: jnp.ndarray, rev_index: jnp.ndarray
) -> dict[str, jnp.ndarray | None]:
    """``GROVEREmbedding.forward``: run the encoder, repackage into a named dict."""
    embedding_output_type = cfg.embedding_output_type
    output = gtrans_encoder_apply(params["encoders"], cfg, cfg.num_attn_head, embedding_output_type, x, edge_index, edge_attr, rev_index)

    if embedding_output_type == "atom":
        return {"atom_from_atom": output[0], "atom_from_bond": output[1], "bond_from_atom": None, "bond_from_bond": None}
    elif embedding_output_type == "bond":
        return {"atom_from_atom": None, "atom_from_bond": None, "bond_from_atom": output[0], "bond_from_bond": output[1]}
    else:  # "both"
        return {
            "atom_from_atom": output[0][0],
            "bond_from_atom": output[0][1],
            "atom_from_bond": output[1][0],
            "bond_from_bond": output[1][1],
        }
