"""Small op appliers (``(params, x) -> y``) + activation dispatcher.
``params`` is a plain nested dict pytree mirroring a checkpoint's saved
structure - see ``convert.py``.
"""


import jax.numpy as jnp

SUPPORTED_ACTIVATIONS = ("ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU", "Linear")


def linear_apply(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """``nn.Linear``: ``y = x @ W.T + b``, ``W`` stored PyTorch-style ``(out, in)``."""
    y = x @ params["weight"].T
    if "bias" in params:
        y = y + params["bias"]
    return y


def layernorm_apply(params: dict, x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """``nn.LayerNorm`` over the last axis, PyTorch's default ``eps=1e-5``, biased variance."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normed = (x - mean) / jnp.sqrt(var + eps)
    return normed * params["weight"] + params["bias"]


def prelu_apply(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """``nn.PReLU()``, single shared slope: ``y = max(0, x) + a * min(0, x)``."""
    a = params["weight"]
    return jnp.where(x >= 0, x, a * x)


def activation_apply(activation: str, params: dict | None, x: jnp.ndarray) -> jnp.ndarray:
    """Dispatch over GROVER's supported activation set."""
    if activation == "ReLU":
        return jnp.maximum(x, 0.0)
    elif activation == "LeakyReLU":
        return jnp.where(x >= 0, x, 0.1 * x)
    elif activation == "PReLU":
        return prelu_apply(params, x)
    elif activation == "tanh":
        return jnp.tanh(x)
    elif activation == "SELU":
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * jnp.where(x >= 0, x, alpha * (jnp.exp(x) - 1.0))
    elif activation == "ELU":
        return jnp.where(x >= 0, x, jnp.exp(x) - 1.0)
    elif activation == "Linear":
        return x
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def activation_has_params(activation: str) -> bool:
    return activation == "PReLU"
