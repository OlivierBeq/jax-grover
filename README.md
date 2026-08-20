<div align="center">

  # 🧬 jax-GROVER

  [![CI](https://github.com/OlivierBeq/jax-grover/actions/workflows/ci.yml/badge.svg)](https://github.com/OlivierBeq/jax-grover/actions/workflows/ci.yml)
  [![Supported Python versions](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![arXiv](https://img.shields.io/badge/arXiv-2007.02835-b31b1b.svg)](https://arxiv.org/abs/2007.02835)

</div>

A JAX reimplementation of [GROVER](https://arxiv.org/abs/2007.02835) (dual-view atom/bond message-passing transformer for molecular representation learning), numerically compatible with the official `grover_base.pt` (hidden 800) and `grover_large.pt` (hidden 1200) checkpoints. Load a pretrained checkpoint and embed SMILES in a few lines, with no PyTorch dependency at inference time.

## ✨ Features

- 🧪 **One-call fingerprinting** — `GroverModel().embed_smiles(smiles)`: checkpoint auto-downloaded, verified, and JIT-warmed on first use.
- 🔀 **Four fingerprint branches** — atom- and bond-view embeddings, individually (`"atom"`, `"bond"`) or concatenated (`"both"`).
- 📦 **Memory-bounded batching** — chunked processing handles arbitrarily large SMILES inputs without blowing up memory.
- 🕸️ **Dependency-free graph batching** — scatter-sum (`segment_sum`) aggregation over a COO `edge_index`/`rev_index` representation, numerically equivalent to GROVER's padded-neighbor-index gather but without a shared "padding row" for batches to leak through.
- 🧱 **Composable, functional core** — every layer of the stack (featurization, batching, encoder, pooling, checkpoint conversion) is a plain function you can call independently of `GroverModel`.
- 🌱 **Trainable from scratch** — random parameter initialization for the full embedding stack, no checkpoint required.
- ✅ **Checkpoint-pinned regression tests** — outputs of the real `grover_base.pt`/`grover_large.pt` checkpoints are checked against an independent PyTorch reference to ~1e-5 max abs diff.

## 📦 Installation

```bash
pip install jax-grover
```

Optional extras enable additional functionality:

| Extra | Enables |
|---|---|
| `jax-grover[convert]` | Reading original PyTorch checkpoints (`torch`) |
| `jax-grover[testing]` | Running the test suite (`pytest`, `torch`) |
| `jax-grover[lint]` | Linting (`ruff`) |

The core library depends only on `jax`, `numpy`, and `rdkit` — `torch` is needed solely for reading `.pt` checkpoint files (it never runs a forward pass) and for the test suite.

## 🛠️ Requirements

- Python 3.11+
- [JAX](https://github.com/google/jax)
- [RDKit](https://www.rdkit.org/docs/Install.html)

## 💡 Usage

### Quickstart

```python
from jax_grover import GroverModel

model = GroverModel()  # downloads grover_large.pt, then JIT warmup
fingerprints = model.embed_smiles(["CCO", "c1ccccc1"])  # (2, hidden_size * 4)
single = model.embed_smiles("CC(=O)Oc1ccccc1C(=O)O")     # (hidden_size * 4,)
```

Reuse one `GroverModel` instance across calls. Key options: `model_type="base"|"large"`, `fingerprint_source="atom"|"bond"|"both"` (settable at construction or per-call), `chunk_size=` on `embed_smiles`, `warmup=False`/`download=False`, or bring your own `params=`/`config=`.

> **Note:** pooling is JIT-compiled and cached per `(config, num_atoms, num_edges, num_graphs)`. Unlike padded-sequence models, GROVER wastes no compute on padding — but exact shape matches across calls are rare for real SMILES batches, so JIT reuse mostly benefits repeated identical inputs. `GroverModel()` compiles once on a trivial molecule at construction so integration errors surface immediately rather than on first real use.

<details>
<summary><strong>Lower-level pieces</strong></summary>

```python
from jax_grover.fingerprint import load_grover_encoder
from jax_grover.graph import smiles_list_to_batch
from jax_grover.embedding import pool_grover_embeddings

params, config = load_grover_encoder(model_type="base")
batch = smiles_list_to_batch(["CCO", "c1ccccc1"])
pooled = pool_grover_embeddings(params, config, batch)
# {"atom_from_atom": ..., "atom_from_bond": ..., "bond_from_atom": ..., "bond_from_bond": ...}
```
</details>

<details>
<summary><strong>Loading a checkpoint file directly</strong></summary>

```python
from jax_grover.convert import load_grover_checkpoint

params, config = load_grover_checkpoint("/path/to/grover_base.pt")
model = GroverModel(params=params, config=config, download=False)
```
</details>

<details>
<summary><strong>Random initialization (training from scratch)</strong></summary>

```python
import jax
from jax_grover.config import GroverConfig
from jax_grover.convert import edge_node_fdim
from jax_grover.init import init_grover_embedding_params

config = GroverConfig(hidden_size=128, depth=6, num_attn_head=4)
edge_fdim, node_fdim = edge_node_fdim()
params = init_grover_embedding_params(jax.random.PRNGKey(0), config, edge_fdim, node_fdim)
```
</details>

## 🎯 Scope

Pretrained **encoder + mean-pooled fingerprint** only — featurization, batching, `GTransEncoder`/`MTBlock`/`Readout`, and checkpoint conversion. This is the only part of GROVER with released weights to reproduce; finetuning heads and pretraining decoders are out of scope.

## ⚡ Performance

Fingerprinting throughput for `grover_large` (`fingerprint_source="both"`), measured on 300 random SMILES (mean length ~41 chars) from a real dataset, chunked at 128 molecules/chunk. **Cold** = first call, including the per-chunk JIT compile (GROVER batches aren't padded, so each distinct atom/edge/graph-count triggers a recompile). **Warm** = steady-state throughput once shapes are already compiled — the realistic number for large batch jobs. CPU figures use `taskset` to restrict the process to N logical cores on a 32-core machine; GPU is a single 8GB RTX 2000 Ada.

| Device | Cores | Cold throughput (mol/s) | Warm throughput (mol/s) | Warm speedup vs 1 CPU core |
|---|---|---|---|---|
| CPU | 1 | 5.1 | 6.3 | 1.0x |
| CPU | 2 | 6.2 | 11.7 | 1.9x |
| CPU | 4 | 14.8 | 20.8 | 3.3x |
| CPU | 8 | 20.0 | 35.3 | 5.7x |
| CPU | 16 | 23.9 | 42.0 | 6.7x |
| CPU | 32 | 25.4 | 49.1 | 7.9x |
| GPU (NVIDIA RTX 2000 Ada, 8GB) | — (1 device) | 8.5 | **567.7** | 90.8x |

Takeaways:
- CPU scaling is close to linear up to ~8 cores, then diminishes sharply (25% parallel efficiency at 32 cores vs. 1) — likely memory-bandwidth bound.
- The GPU's steady-state throughput is ~11.5x the 32-core CPU figure, but its cold-start is *slower* than CPU (higher XLA codegen/launch overhead per compile) — CPU can respond faster for one-off/low-volume calls, GPU wins once compilation is amortized over a large batch.
- SMILES parsing/graph-building (RDKit, not JAX) is cheap by comparison — over 1,600 mol/s single-threaded — and is not the bottleneck versus the model forward pass.

## ✅ Correctness

Verified at four levels, from raw ops up to the public API:

| Test file | Verifies |
|---|---|
| `test_features.py`, `test_batching.py` | Featurization and aggregation ops, against frozen fixtures / naive reimplementations |
| `test_encoder_equivalence.py` | Architectural invariants (shapes, determinism, batch-composition independence) on random-init models |
| `test_real_checkpoints.py` | Real `grover_base.pt`/`grover_large.pt`, against a golden fixture from an independent PyTorch reference (~1e-5 max abs diff — float32 noise), pinned as a regression guard |
| `test_fingerprint.py` | `embed_smiles`, against independent numpy mean-pooling |

Checkpoint-dependent tests skip automatically when the weights aren't present locally.

## 📥 Getting the pretrained checkpoints

Hosted as assets on this repo's ["Model weights" GitHub Release](https://github.com/OlivierBeq/jax-grover/releases/tag/model_weights) (`grover_base.pt` ~190MB, `grover_large.pt` ~430MB — gitignored, not tracked in-repo).

> While this repo is private, unauthenticated asset downloads 404. Works for anyone once the repo is public; until then, only in environments already authenticated against it.

Downloaded automatically (checksum-verified) on first `GroverModel()` / `load_grover_encoder` / `embed_smiles` call, or explicitly:

```python
from jax_grover.weights.download_weights import download_grover_weights

download_grover_weights(base=True, large=False)  # -> src/jax_grover/weights/grover_base.pt
```

The download is stdlib-only (`urllib`), atomic (temp file + rename), and sha256-verified. Override the source per size via environment variables:

```bash
export JAX_GROVER_WEIGHTS_URL_BASE="https://.../grover_base.pt"
export JAX_GROVER_WEIGHTS_URL_LARGE="https://.../grover_large.pt"
```

Or supply your own checkpoint (`GroverModel(params=..., config=..., download=False)` or `checkpoint_path=...`) in the original GROVER format: a `torch.save`'d dict with `"args"` + `"state_dict"` keys (`torch` only reads the file, never computes).

## 🖋️ Citation

This repository is an independent JAX port; if you use GROVER in your research, please cite the original paper:

```bibtex
@inproceedings{rong2020grover,
  title={Self-Supervised Graph Transformer on Large-Scale Molecular Data},
  author={Rong, Yu and Bian, Yatao and Xu, Tingyang and Xie, Weiyang and Wei, Ying and Huang, Wenbing and Huang, Junzhou},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).
