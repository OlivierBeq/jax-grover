"""Download + load the pretrained GROVER checkpoints from this repository's
own GitHub Release (tag ``model_weights``)."""
from __future__ import annotations

import os
from pathlib import Path

from .._download import download_file, sha256sum  # noqa: F401 (sha256sum re-exported for callers)
from ..config import GroverConfig

__all__ = [
    "weights_dir",
    "default_weights_path",
    "download_default_weights",
    "download_grover_weights",
    "load_default_model",
    "weights_path",
]

# Can be overridden per model type without a code change via the
# JAX_GROVER_WEIGHTS_URL_{BASE,LARGE} environment variables, which take
# precedence over these constants.
_RELEASE_BASE_URL = "https://github.com/OlivierBeq/jax-grover/releases/download/model_weights"

_DEFAULT_URLS = {
    "base": f"{_RELEASE_BASE_URL}/grover_base.pt",
    "large": f"{_RELEASE_BASE_URL}/grover_large.pt",
}
_DEFAULT_SHA256 = {
    "base": "47e095880d71baf29ea6f6253473cd56d5406213fa82959c6e14ea469e06b1de",
    "large": "4b0c436fbd6ed8539fa92a0c9f890878f5e3dd8591d959315e773efb6302baaa",
}
_URL_ENV_VARS = {
    "base": "JAX_GROVER_WEIGHTS_URL_BASE",
    "large": "JAX_GROVER_WEIGHTS_URL_LARGE",
}


def _check_model_type(model_type: str) -> None:
    if model_type not in _DEFAULT_URLS:
        raise ValueError(f"model_type must be one of {sorted(_DEFAULT_URLS)}, got {model_type!r}")


def weights_dir() -> Path:
    return Path(__file__).resolve().parent


def default_weights_path(model_type: str) -> Path:
    _check_model_type(model_type)
    return weights_dir() / f"grover_{model_type}.pt"


# Backwards-compatible alias.
def weights_path(model_type: str) -> str:
    return str(default_weights_path(model_type))


def download_default_weights(
    model_type: str,
    url: str | None = None,
    dest: str | Path | None = None,
    expected_sha256: str | None = None,
    force: bool = False,
) -> Path:
    """Download one checkpoint ("base"/"large") from the GitHub release to
    ``dest`` (default ``default_weights_path(model_type)``). No-op if
    ``dest`` exists, unless ``force``. ``url``/``expected_sha256`` override
    the release URL / known hash; ``url`` also falls back to
    ``JAX_GROVER_WEIGHTS_URL_{BASE,LARGE}``.
    """
    _check_model_type(model_type)
    dest = Path(dest) if dest is not None else default_weights_path(model_type)
    resolved_url = url or os.environ.get(_URL_ENV_VARS[model_type]) or _DEFAULT_URLS[model_type]
    if not dest.exists() and not resolved_url:
        raise RuntimeError(
            f"No checkpoint found at {dest}, and no download URL is configured. "
            f"Either set the {_URL_ENV_VARS[model_type]} environment variable, "
            "pass url=..., or place the file there yourself."
        )
    return download_file(
        resolved_url,
        dest,
        expected_sha256=expected_sha256 if expected_sha256 is not None else _DEFAULT_SHA256[model_type],
        force=force,
        label=f"pretrained GROVER {model_type} weights",
    )


def download_grover_weights(base: bool = True, large: bool = False, progress: bool = False, force: bool = False) -> bool:
    """Download one or both model sizes at once (``progress`` kept for
    backwards compatibility; downloads always print progress)."""
    if not (base or large):
        raise ValueError("At least one of base/large must be selected.")
    if base:
        download_default_weights("base", force=force)
    if large:
        download_default_weights("large", force=force)
    return True


def load_default_model(model_type: str = "large", download: bool = True) -> tuple[dict, GroverConfig]:
    """Load a pretrained checkpoint by name, converted to a JAX param pytree
    + ``GroverConfig``. Downloads it first if missing and ``download`` is True.
    """
    from ..convert import load_grover_checkpoint  # local import: avoids a cycle with convert.py

    path = default_weights_path(model_type)
    if download:
        path = download_default_weights(model_type)
    elif not path.exists():
        raise FileNotFoundError(f"{path} not found locally and download=False - either download it first or pass download=True.")
    return load_grover_checkpoint(str(path))
