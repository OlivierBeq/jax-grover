"""Random from-scratch param pytree init, matching GROVER's own scheme:
1-D params (biases, norms, PReLU slope) -> 0; 2+-D params -> Xavier-normal.
"""

import math

import jax
import jax.numpy as jnp

from .config import GroverConfig


class _KeySequence:
    """Deterministic stream of PRNG subkeys split off a single root key."""

    def __init__(self, key: jax.Array):
        self._key = key

    def next(self) -> jax.Array:
        self._key, sub = jax.random.split(self._key)
        return sub


def _xavier_normal(key: jax.Array, out_dim: int, in_dim: int) -> jnp.ndarray:
    std = math.sqrt(2.0 / (in_dim + out_dim))
    return std * jax.random.normal(key, (out_dim, in_dim), dtype=jnp.float32)


def _init_linear(keys: _KeySequence, in_dim: int, out_dim: int, has_bias: bool) -> dict:
    params = {"weight": _xavier_normal(keys.next(), out_dim, in_dim)}
    if has_bias:
        params["bias"] = jnp.zeros((out_dim,), dtype=jnp.float32)
    return params


def _init_layernorm(dim: int) -> dict:
    return {"weight": jnp.zeros((dim,), dtype=jnp.float32), "bias": jnp.zeros((dim,), dtype=jnp.float32)}


def _init_act(cfg: GroverConfig) -> dict | None:
    if cfg.activation != "PReLU":
        return None
    return {"weight": jnp.zeros((1,), dtype=jnp.float32)}


def _init_mpn_encoder(keys: _KeySequence, cfg: GroverConfig, input_dim: int | None) -> dict:
    params: dict = {}
    if input_dim is not None:
        params["W_i"] = _init_linear(keys, input_dim, cfg.hidden_size, cfg.bias)
    params["W_h"] = _init_linear(keys, cfg.hidden_size, cfg.hidden_size, cfg.bias)
    act = _init_act(cfg)
    if act is not None:
        params["act_func"] = act
    return params


def _init_head(keys: _KeySequence, cfg: GroverConfig) -> dict:
    return {
        "mpn_q": _init_mpn_encoder(keys, cfg, None),
        "mpn_k": _init_mpn_encoder(keys, cfg, None),
        "mpn_v": _init_mpn_encoder(keys, cfg, None),
    }


def _init_mt_block(keys: _KeySequence, cfg: GroverConfig, input_dim: int) -> dict:
    params: dict = {"W_i": _init_linear(keys, input_dim, cfg.hidden_size, cfg.bias)}
    act = _init_act(cfg)
    if act is not None:
        params["act_func"] = act
    params["layernorm"] = _init_layernorm(cfg.hidden_size)
    # Q/K/V linear_layers always carry bias (independent of cfg.bias).
    params["attn"] = {
        "linear_layers": [_init_linear(keys, cfg.hidden_size, cfg.hidden_size, True) for _ in range(3)],
        "output_linear": _init_linear(keys, cfg.hidden_size, cfg.hidden_size, cfg.bias),
    }
    params["W_o"] = _init_linear(keys, cfg.hidden_size * cfg.num_attn_head, cfg.hidden_size, cfg.bias)
    params["sublayer"] = {"norm": _init_layernorm(cfg.hidden_size)}
    params["heads"] = [_init_head(keys, cfg) for _ in range(cfg.num_attn_head)]
    return params


def _init_ffn(keys: _KeySequence, cfg: GroverConfig, in_dim: int) -> dict:
    d_ff = cfg.hidden_size * 4
    params = {
        "W_1": _init_linear(keys, in_dim, d_ff, True),
        "W_2": _init_linear(keys, d_ff, cfg.hidden_size, True),
    }
    act = _init_act(cfg)
    if act is not None:
        params["act_func"] = act
    return params


def init_grover_embedding_params(key: jax.Array, cfg: GroverConfig, edge_fdim: int, node_fdim: int) -> dict:
    """Random param pytree, same structure as ``convert.params_from_state_dict``."""
    keys = _KeySequence(key)

    encoders = {
        "node_blocks": [_init_mt_block(keys, cfg, node_fdim if i == 0 else cfg.hidden_size) for i in range(cfg.num_mt_block)],
        "edge_blocks": [_init_mt_block(keys, cfg, edge_fdim if i == 0 else cfg.hidden_size) for i in range(cfg.num_mt_block)],
        "ffn_atom_from_atom": _init_ffn(keys, cfg, cfg.hidden_size + node_fdim),
        "ffn_atom_from_bond": _init_ffn(keys, cfg, cfg.hidden_size + node_fdim),
        "ffn_bond_from_atom": _init_ffn(keys, cfg, cfg.hidden_size + edge_fdim),
        "ffn_bond_from_bond": _init_ffn(keys, cfg, cfg.hidden_size + edge_fdim),
        "atom_from_atom_sublayer": {"norm": _init_layernorm(cfg.hidden_size)},
        "atom_from_bond_sublayer": {"norm": _init_layernorm(cfg.hidden_size)},
        "bond_from_atom_sublayer": {"norm": _init_layernorm(cfg.hidden_size)},
        "bond_from_bond_sublayer": {"norm": _init_layernorm(cfg.hidden_size)},
    }
    return {"encoders": encoders}
