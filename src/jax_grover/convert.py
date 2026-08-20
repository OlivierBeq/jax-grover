"""Convert a released GROVER checkpoint's saved parameters into this
package's nested-dict JAX param pytree.

A checkpoint is a ``torch.save``'d dict with an ``"args"`` config namespace
and a ``"state_dict"`` whose encoder keys look like
``grover.encoders.node_blocks.0.W_i.weight``. Conversion strips the
``grover.`` prefix, then walks the known static module hierarchy (see
``params_from_state_dict``). ``load_grover_checkpoint`` does this directly
from a ``.pt`` path (``torch`` only reads the file, never computes).
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from .config import GroverConfig
from .features import atom_feature_dim, bond_feature_dim

StateDict = dict[str, Any]


def _to_jnp(value: Any) -> jnp.ndarray:
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu().numpy()
    return jnp.asarray(np.asarray(value))


def _linear_params(sd: StateDict, prefix: str, has_bias: bool) -> dict:
    params = {"weight": _to_jnp(sd[f"{prefix}.weight"])}
    bias_key = f"{prefix}.bias"
    if has_bias and bias_key in sd:
        params["bias"] = _to_jnp(sd[bias_key])
    return params


def _layernorm_params(sd: StateDict, prefix: str) -> dict:
    return {"weight": _to_jnp(sd[f"{prefix}.weight"]), "bias": _to_jnp(sd[f"{prefix}.bias"])}


def _act_params(sd: StateDict, cfg: GroverConfig, prefix: str) -> dict | None:
    if cfg.activation != "PReLU":
        return None
    key = f"{prefix}.weight"
    if key not in sd:
        return None
    return {"weight": _to_jnp(sd[key])}


def _mpn_encoder_params(sd: StateDict, cfg: GroverConfig, prefix: str, input_layer: str) -> dict:
    params: dict = {}
    if input_layer == "fc":
        params["W_i"] = _linear_params(sd, f"{prefix}.W_i", cfg.bias)
    params["W_h"] = _linear_params(sd, f"{prefix}.W_h", cfg.bias)
    act = _act_params(sd, cfg, f"{prefix}.act_func")
    if act is not None:
        params["act_func"] = act
    return params


def _head_params(sd: StateDict, cfg: GroverConfig, prefix: str) -> dict:
    return {
        "mpn_q": _mpn_encoder_params(sd, cfg, f"{prefix}.mpn_q", "none"),
        "mpn_k": _mpn_encoder_params(sd, cfg, f"{prefix}.mpn_k", "none"),
        "mpn_v": _mpn_encoder_params(sd, cfg, f"{prefix}.mpn_v", "none"),
    }


def _mt_block_params(sd: StateDict, cfg: GroverConfig, prefix: str) -> dict:
    params: dict = {"W_i": _linear_params(sd, f"{prefix}.W_i", cfg.bias)}
    act = _act_params(sd, cfg, f"{prefix}.act_func")
    if act is not None:
        params["act_func"] = act
    params["layernorm"] = _layernorm_params(sd, f"{prefix}.layernorm")
    # Q/K/V linear_layers always carry bias regardless of cfg.bias (quirk of
    # the original architecture); output_linear follows cfg.bias.
    params["attn"] = {
        "linear_layers": [_linear_params(sd, f"{prefix}.attn.linear_layers.{i}", True) for i in range(3)],
        "output_linear": _linear_params(sd, f"{prefix}.attn.output_linear", cfg.bias),
    }
    params["W_o"] = _linear_params(sd, f"{prefix}.W_o", cfg.bias)
    params["sublayer"] = {"norm": _layernorm_params(sd, f"{prefix}.sublayer.norm")}
    params["heads"] = [_head_params(sd, cfg, f"{prefix}.heads.{i}") for i in range(cfg.num_attn_head)]
    return params


def _ffn_params(sd: StateDict, cfg: GroverConfig, prefix: str) -> dict:
    params = {
        "W_1": _linear_params(sd, f"{prefix}.W_1", True),
        "W_2": _linear_params(sd, f"{prefix}.W_2", True),
    }
    act = _act_params(sd, cfg, f"{prefix}.act_func")
    if act is not None:
        params["act_func"] = act
    return params


def params_from_state_dict(sd: StateDict, cfg: GroverConfig) -> dict:
    """Build the param pytree from a bare ``state_dict`` (keys like
    ``encoders.node_blocks.0.W_i.weight``, no ``grover.`` prefix)."""
    encoders = {
        "node_blocks": [_mt_block_params(sd, cfg, f"encoders.node_blocks.{i}") for i in range(cfg.num_mt_block)],
        "edge_blocks": [_mt_block_params(sd, cfg, f"encoders.edge_blocks.{i}") for i in range(cfg.num_mt_block)],
        "ffn_atom_from_atom": _ffn_params(sd, cfg, "encoders.ffn_atom_from_atom"),
        "ffn_atom_from_bond": _ffn_params(sd, cfg, "encoders.ffn_atom_from_bond"),
        "ffn_bond_from_atom": _ffn_params(sd, cfg, "encoders.ffn_bond_from_atom"),
        "ffn_bond_from_bond": _ffn_params(sd, cfg, "encoders.ffn_bond_from_bond"),
        "atom_from_atom_sublayer": {"norm": _layernorm_params(sd, "encoders.atom_from_atom_sublayer.norm")},
        "atom_from_bond_sublayer": {"norm": _layernorm_params(sd, "encoders.atom_from_bond_sublayer.norm")},
        "bond_from_atom_sublayer": {"norm": _layernorm_params(sd, "encoders.bond_from_atom_sublayer.norm")},
        "bond_from_bond_sublayer": {"norm": _layernorm_params(sd, "encoders.bond_from_bond_sublayer.norm")},
    }
    return {"encoders": encoders}


# Checkpoint "args" attribute names mapping 1:1 onto GroverConfig fields.
_CONFIG_FIELD_NAMES = (
    "hidden_size", "depth", "dropout", "activation", "bias", "dense", "undirected", "num_mt_block",
    "num_attn_head", "embedding_output_type", "backbone", "res_connection", "self_attention", "attn_hidden",
    "attn_out", "dist_coff", "bond_drop_rate", "fine_tune_coff", "ffn_hidden_size", "ffn_num_layers",
)


def grover_config_from_old_args(old_args: Any) -> GroverConfig:
    kwargs = {}
    for name in _CONFIG_FIELD_NAMES:
        if hasattr(old_args, name):
            kwargs[name] = getattr(old_args, name)
    return GroverConfig(**kwargs)


def load_old_checkpoint(path: str):
    """Read a released GROVER ``.pt`` checkpoint. Requires ``torch``."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["args"], state["state_dict"]


def load_grover_checkpoint(path: str) -> tuple[dict, GroverConfig]:
    """Load a released ``.pt`` checkpoint into a JAX param pytree + ``GroverConfig``."""
    old_args, old_state_dict = load_old_checkpoint(path)
    config = grover_config_from_old_args(old_args)
    stripped = {k[len("grover.") :]: v for k, v in old_state_dict.items() if k.startswith("grover.")}
    params = params_from_state_dict(stripped, config)
    return params, config


def edge_node_fdim() -> tuple[int, int]:
    edge_fdim = bond_feature_dim() + atom_feature_dim()
    node_fdim = atom_feature_dim()
    return edge_fdim, node_fdim
