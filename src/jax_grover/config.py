"""GROVER encoder config. Field names match a released checkpoint's saved
args 1:1 (see ``convert.grover_config_from_old_args``). Training-only fields
(``dropout``, ``bond_drop_rate``, ...) are no-ops at inference.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GroverConfig:
    hidden_size: int = 800
    depth: int = 3
    dropout: float = 0.0
    activation: str = "PReLU"
    bias: bool = False
    dense: bool = False
    # Keep True for checkpoint-loaded inference: matches the (inverted)
    # `if self.dense` branch the released checkpoints were trained under.
    # See layers.mpn_encoder_apply.
    faithful_dense_bug: bool = True
    undirected: bool = False
    num_mt_block: int = 1
    num_attn_head: int = 4
    embedding_output_type: Literal["atom", "bond", "both"] = "both"
    backbone: str = "gtrans"
    res_connection: bool = False
    self_attention: bool = False
    attn_hidden: int = 4
    attn_out: int = 128
    dist_coff: float = 0.1
    bond_drop_rate: float = 0.0
    fine_tune_coff: float = 1.0
    ffn_hidden_size: int | None = None
    ffn_num_layers: int = 2

    def __post_init__(self):
        if self.ffn_hidden_size is None:
            object.__setattr__(self, "ffn_hidden_size", self.hidden_size)
