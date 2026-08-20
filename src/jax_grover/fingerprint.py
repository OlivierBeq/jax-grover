"""Stateless functional API: SMILES -> fixed-size GROVER fingerprints.
See ``inference.GroverModel`` for the recommended object-oriented entry
point; these are its lower-level building blocks.
"""

from typing import Literal

import jax.numpy as jnp

from .config import GroverConfig
from .convert import load_grover_checkpoint
from .embedding import ATOM_BRANCHES, BOND_BRANCHES, pool_grover_embeddings
from .graph import smiles_list_to_batch

ModelType = Literal["base", "large"]
FingerprintSource = Literal["atom", "bond", "both"]


def load_grover_encoder(model_type: ModelType = "large", download: bool = True) -> tuple[dict, GroverConfig]:
    """Load a pretrained GROVER checkpoint by name ("base"/"large"), converted
    to a JAX param pytree + ``GroverConfig``. Downloads it if missing and
    ``download`` is True."""
    from .weights.download_weights import load_default_model

    return load_default_model(model_type=model_type, download=download)


def grover_fingerprint(params: dict, cfg: GroverConfig, batch, fingerprint_source: FingerprintSource = "both") -> jnp.ndarray:
    """Encoder + mean-pool readout -> one vector per molecule (branches
    concatenated in ``ATOM_BRANCHES + BOND_BRANCHES`` order for "both")."""
    if fingerprint_source == "atom":
        branches = ATOM_BRANCHES
    elif fingerprint_source == "bond":
        branches = BOND_BRANCHES
    elif fingerprint_source == "both":
        branches = ATOM_BRANCHES + BOND_BRANCHES
    else:
        raise ValueError("fingerprint_source must be one of ['atom', 'bond', 'both']")

    pooled = pool_grover_embeddings(params, cfg, batch)
    return jnp.concatenate([pooled[b] for b in branches], axis=1)


def embed_smiles(
    smiles_list: list[str],
    model_type: ModelType = "large",
    checkpoint_path: str | None = None,
    download: bool = True,
    fingerprint_source: FingerprintSource = "both",
) -> jnp.ndarray:
    """One-shot: SMILES -> fixed-size GROVER fingerprints. Loads
    ``grover_{model_type}.pt`` unless ``checkpoint_path`` is given. For
    repeated calls, prefer ``inference.GroverModel`` (this reloads the
    checkpoint from disk every call)."""
    if checkpoint_path is not None:
        params, cfg = load_grover_checkpoint(checkpoint_path)
    else:
        params, cfg = load_grover_encoder(model_type=model_type, download=download)
    batch = smiles_list_to_batch(smiles_list)
    return grover_fingerprint(params, cfg, batch, fingerprint_source=fingerprint_source)
