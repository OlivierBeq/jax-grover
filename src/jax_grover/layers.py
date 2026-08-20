"""Dual-view GTransformer encoder building blocks: message-passing +
attention as pure ``(params, ..., x) -> y`` functions. Inference-only: no
dropout, no dynamic-depth sampling (both training-only, inert at eval time).
"""


import jax.nn as jnn
import jax.numpy as jnp

from .batch_ops import aggregate_atom_view, aggregate_bond_view, aggregate_edge_attr_to_atom, segment_mean
from .config import GroverConfig
from .nn_utils import activation_apply, layernorm_apply, linear_apply


def positionwise_feedforward_apply(params: dict, cfg: GroverConfig, x: jnp.ndarray) -> jnp.ndarray:
    h = linear_apply(params["W_1"], x)
    h = activation_apply(cfg.activation, params.get("act_func"), h)
    return linear_apply(params["W_2"], h)


def sublayer_connection_apply(params: dict, inputs: jnp.ndarray | None, outputs: jnp.ndarray) -> jnp.ndarray:
    """Residual connection + layer norm (norm applied to the sublayer output)."""
    normed = layernorm_apply(params["norm"], outputs)
    if inputs is None:
        return normed
    return inputs + normed


def scaled_dot_product_attention(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    d_k = query.shape[-1]
    scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(jnp.asarray(d_k, dtype=query.dtype))
    p_attn = jnn.softmax(scores, axis=-1)
    return jnp.matmul(p_attn, value)


def multi_headed_attention_apply(params: dict, h: int, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """``query``/``key``/``value``: ``(N, seq, d_model)``. ``h`` splits
    ``d_model`` into ``h`` inner heads - independent from the outer ``seq``
    dim, which comes from GROVER's per-``Head`` Q/K/V (see ``MTBlock``)."""
    n = query.shape[0]
    seq = query.shape[1]
    d_model = query.shape[-1]
    d_k = d_model // h

    def project(p: dict, x: jnp.ndarray) -> jnp.ndarray:
        y = linear_apply(p, x)
        # seq given explicitly, not -1: reshape with n == 0 (batch with zero
        # total edges) and an inferred dim is ambiguous in numpy/JAX/PyTorch alike.
        y = y.reshape(n, seq, h, d_k)
        return jnp.transpose(y, (0, 2, 1, 3))  # (N, h, seq, d_k)

    q = project(params["linear_layers"][0], query)
    k = project(params["linear_layers"][1], key)
    v = project(params["linear_layers"][2], value)

    x = scaled_dot_product_attention(q, k, v)  # (N, h, seq, d_k)
    x = jnp.transpose(x, (0, 2, 1, 3)).reshape(n, seq, h * d_k)  # (N, seq, d_model)
    return linear_apply(params["output_linear"], x)


def mpn_encoder_apply(
    params: dict,
    cfg: GroverConfig,
    atom_messages: bool,
    attach_fea: bool,
    input_layer: str,
    init_messages: jnp.ndarray,
    edge_index: jnp.ndarray,
    rev_index: jnp.ndarray,
    num_atoms: int,
    attached_features: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if input_layer == "fc":
        input_ = linear_apply(params["W_i"], init_messages)
        message = activation_apply(cfg.activation, params.get("act_func"), input_)
    else:
        input_ = init_messages
        message = input_

    src = edge_index[0]

    for _ in range(cfg.depth - 1):
        if cfg.undirected:
            if atom_messages:
                raise NotImplementedError("undirected=True with atom_messages=True is not a valid configuration.")
            message = (message + message[rev_index]) / 2

        if atom_messages:
            nei_message = aggregate_atom_view(message, edge_index, num_atoms)
            if attach_fea:
                attached_nei = aggregate_edge_attr_to_atom(attached_features, edge_index, num_atoms)
                message = jnp.concatenate([nei_message, attached_nei], axis=1)
            else:
                message = nei_message
        else:
            if not attach_fea:
                message = aggregate_bond_view(message, edge_index, rev_index, num_atoms)
            else:
                nei_message = aggregate_edge_attr_to_atom(message, edge_index, num_atoms)
                attached_nei = aggregate_atom_view(attached_features, edge_index, num_atoms)
                a_message = jnp.concatenate([nei_message, attached_nei], axis=1)
                rev_message = jnp.concatenate([message[rev_index], attached_features[src[rev_index]]], axis=1)
                message = a_message[src] - rev_message

        message = linear_apply(params["W_h"], message)

        # Inverted vs. what the name implies - see GroverConfig.faithful_dense_bug.
        if cfg.faithful_dense_bug:
            if cfg.dense:
                message = activation_apply(cfg.activation, params.get("act_func"), message)
            else:
                message = activation_apply(cfg.activation, params.get("act_func"), input_ + message)
        else:
            if not cfg.dense:
                message = activation_apply(cfg.activation, params.get("act_func"), message)
            else:
                message = activation_apply(cfg.activation, params.get("act_func"), input_ + message)

    return message


def head_apply(
    params: dict, cfg: GroverConfig, atom_messages: bool, hidden: jnp.ndarray, edge_index: jnp.ndarray, rev_index: jnp.ndarray, num_atoms: int
):
    """One head: Q/K/V from three independent MPNEncoders."""
    kwargs = dict(
        cfg=cfg,
        atom_messages=atom_messages,
        attach_fea=False,
        input_layer="none",
        init_messages=hidden,
        edge_index=edge_index,
        rev_index=rev_index,
        num_atoms=num_atoms,
    )
    q = mpn_encoder_apply(params["mpn_q"], **kwargs)
    k = mpn_encoder_apply(params["mpn_k"], **kwargs)
    v = mpn_encoder_apply(params["mpn_v"], **kwargs)
    return q, k, v


def mt_block_apply(
    params: dict,
    cfg: GroverConfig,
    num_attn_head: int,
    atom_messages: bool,
    hidden: jnp.ndarray,
    edge_index: jnp.ndarray,
    rev_index: jnp.ndarray,
    num_atoms: int,
) -> jnp.ndarray:
    """One "GTransBlock": message-passing heads feed a multi-head attention."""
    if hidden.shape[-1] != cfg.hidden_size:
        hidden = linear_apply(params["W_i"], hidden)
        hidden = activation_apply(cfg.activation, params.get("act_func"), hidden)
        hidden = layernorm_apply(params["layernorm"], hidden)

    queries, keys, values = [], [], []
    for head_params in params["heads"]:
        q, k, v = head_apply(head_params, cfg, atom_messages, hidden, edge_index, rev_index, num_atoms)
        queries.append(q[:, None, :])
        keys.append(k[:, None, :])
        values.append(v[:, None, :])
    queries = jnp.concatenate(queries, axis=1)
    keys = jnp.concatenate(keys, axis=1)
    values = jnp.concatenate(values, axis=1)

    x_out = multi_headed_attention_apply(params["attn"], num_attn_head, queries, keys, values)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1] * x_out.shape[2])  # not -1: see multi_headed_attention_apply
    x_out = linear_apply(params["W_o"], x_out)

    x_in = hidden if cfg.res_connection else None
    return sublayer_connection_apply(params["sublayer"], x_in, x_out)


def readout_mean_apply(embeddings: jnp.ndarray, batch_index: jnp.ndarray, num_graphs: int) -> jnp.ndarray:
    """Node/edge embeddings -> graph-level embeddings via mean-pooling
    (``Readout(rtype="mean")``). Empty group -> zero vector."""
    return segment_mean(embeddings, batch_index, num_graphs)
