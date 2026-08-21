"""High-level inference API: GroverModel, SMILES -> fingerprint. Wraps the
pretrained encoder + JIT-compiled pooling, chunking large inputs internally.
"""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Literal

import numpy as np

from .config import GroverConfig
from .fingerprint import ModelType, grover_fingerprint, load_grover_encoder
from .graph import pad_batch, smiles_list_to_batch

FingerprintSource = Literal["atom", "bond", "both"]

# Molecules per internal chunk; bounds peak memory / XLA program size.
DEFAULT_CHUNK_SIZE = 256


class GroverModel:
    """Loads a pretrained GROVER encoder and exposes SMILES -> fingerprint.

    Example::

        model = GroverModel()  # downloads grover_large.pt on first use
        fps = model.embed_smiles(["CCO", "c1ccccc1"])
    """

    def __init__(
        self,
        model_type: ModelType = "large",
        params: dict | None = None,
        config: GroverConfig | None = None,
        fingerprint_source: FingerprintSource = "both",
        download: bool = True,
        warmup: bool = True,
        n_jobs: int = 1,
    ):
        """
        :param model_type: "base" (hidden_size=800) or "large" (hidden_size=1200).
            Ignored if ``params``/``config`` are given.
        :param params: pre-loaded JAX param pytree; if omitted, loads (and
            downloads if needed) the pretrained ``model_type`` checkpoint.
        :param config: ``GroverConfig`` matching ``params``; required together.
        :param fingerprint_source: default branch for ``embed_smiles``,
            overridable per call.
        :param download: download the checkpoint if missing locally.
        :param warmup: eagerly JIT-compile on a trivial molecule at
            construction, so integration errors surface immediately.
        :param n_jobs: worker processes for SMILES->graph preprocessing. ``1``
            (default) runs serially; ``-1`` uses all CPU cores; other values
            are used as the worker count. Reused across calls; ``close()`` it
            (or use as a context manager) when done.
        """
        if (params is None) != (config is None):
            raise ValueError("params and config must be given together, or not at all.")
        if params is None:
            params, config = load_grover_encoder(model_type=model_type, download=download)
        if n_jobs != -1 and n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 or a positive int, got {n_jobs}")

        self.model_type = model_type
        self.params = params
        self.config = config
        self.fingerprint_source = fingerprint_source
        self.n_jobs = n_jobs
        # "spawn" not "fork": forking a process with JAX's runtime already
        # loaded can deadlock. Workers only run plain RDKit/numpy code.
        self._executor = (
            ProcessPoolExecutor(max_workers=None if n_jobs == -1 else n_jobs, mp_context=multiprocessing.get_context("spawn"))
            if n_jobs != 1
            else None
        )

        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        self.embed_smiles(["CC"])  # smallest bonded molecule

    def close(self) -> None:
        """Shut down the preprocessing worker pool (no-op if ``n_jobs=1``)."""
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> GroverModel:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def embed_smiles(
        self,
        smiles: str | list[str],
        fingerprint_source: FingerprintSource | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> np.ndarray:
        """Embed one or more SMILES into GROVER fingerprint(s), chunked
        internally by ``chunk_size``. Returns ``[fingerprint_dim]`` for a
        single string, ``[n, fingerprint_dim]`` for a list.

        Each chunk is padded to a bucketed shape (``graph.pad_batch``)
        before the JIT-compiled encoder, then sliced back. Keeps compiled
        shapes few and stable across calls sharing ``chunk_size``, avoiding
        a recompile per chunk, with identical output to running unpadded.
        """
        source = fingerprint_source if fingerprint_source is not None else self.fingerprint_source
        single = isinstance(smiles, str)
        smiles_list = [smiles] if single else list(smiles)
        n = len(smiles_list)

        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        parts = []
        for start in range(0, n, chunk_size):
            chunk = smiles_list[start : start + chunk_size]
            batch = smiles_list_to_batch(chunk, executor=self._executor)
            padded = pad_batch(batch, num_graphs=chunk_size + 1)
            fp = grover_fingerprint(self.params, self.config, padded, fingerprint_source=source)
            parts.append(np.asarray(fp)[: len(chunk)])

        result = np.concatenate(parts, axis=0)
        return result[0] if single else result
