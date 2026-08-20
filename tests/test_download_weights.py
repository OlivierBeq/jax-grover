"""Unit tests for download_weights.py that don't touch the network: path
resolution and sha256 verification against local checkpoint files."""

from pathlib import Path

import pytest

from jax_grover.weights.download_weights import _DEFAULT_SHA256, default_weights_path, sha256sum, weights_dir, weights_path


def test_weights_path_matches_package_directory():
    for model_type in ("base", "large"):
        path = default_weights_path(model_type)
        assert path.parent == weights_dir()
        assert path.name == f"grover_{model_type}.pt"
        assert Path(weights_path(model_type)) == path  # backwards-compatible str-returning alias


def test_weights_path_rejects_unknown_model_type():
    with pytest.raises(ValueError):
        default_weights_path("medium")


@pytest.mark.parametrize("model_type", ["base", "large"])
def test_local_checkpoint_hash_matches_release_asset_if_present(model_type):
    path = default_weights_path(model_type)
    if not path.is_file():
        pytest.skip(f"{path} not present locally")
    assert sha256sum(str(path)).lower() == _DEFAULT_SHA256[model_type].lower()
