"""Molecular graph representation + batching, plain numpy (no PyTorch
Geometric). Bonds are added as consecutive ``(a1->a2, a2->a1)`` reverse
pairs; ``edge_attr`` is bond-only (concatenated with source-atom features
lazily, see ``encoder.bond_init_message``).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from .features import MolFeatureContext, atom_feature_dim, atom_features, bond_feature_dim, bond_features


@dataclass
class MolData:
    """One molecule's graph.

    x: (n_atoms, 151) atom features. edge_index: (2, n_directed_edges),
    row 0 = source, row 1 = target; edge ids 2k/2k+1 are mutual reverses.
    edge_attr: (n_directed_edges, 14) bond-only features. rev_index[b]: id
    of edge b's reverse edge.
    """

    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    rev_index: np.ndarray
    smiles: str | None = None


def mol_to_data(mol: Chem.rdchem.Mol, smiles: str | None = None) -> MolData:
    """Build a ``MolData`` from an already-parsed RDKit mol."""
    context = MolFeatureContext(mol)
    n_atoms = mol.GetNumAtoms()

    x = np.zeros((n_atoms, atom_feature_dim()), dtype=np.float32)
    for atom in mol.GetAtoms():
        x[atom.GetIdx()] = atom_features(atom, context)

    edge_index_list = []
    edge_attr_list = []
    rev_index_list = []
    n_bonds = 0
    for a1 in range(n_atoms):
        for a2 in range(a1 + 1, n_atoms):
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is None:
                continue

            fbond = bond_features(bond)
            b1 = n_bonds
            b2 = b1 + 1

            edge_index_list.append((a1, a2))  # b1 = a1 --> a2
            edge_index_list.append((a2, a1))  # b2 = a2 --> a1
            edge_attr_list.append(fbond)
            edge_attr_list.append(fbond)
            rev_index_list.append(b2)
            rev_index_list.append(b1)
            n_bonds += 2

    if n_bonds > 0:
        edge_index = np.array(edge_index_list, dtype=np.int32).T
        edge_attr = np.array(edge_attr_list, dtype=np.float32)
        rev_index = np.array(rev_index_list, dtype=np.int32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int32)
        edge_attr = np.zeros((0, bond_feature_dim()), dtype=np.float32)
        rev_index = np.zeros((0,), dtype=np.int32)

    return MolData(x=x, edge_index=edge_index, edge_attr=edge_attr, rev_index=rev_index, smiles=smiles)


def smiles_to_data(smiles: str) -> MolData:
    """Parse a SMILES string and build its ``MolData``."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    return mol_to_data(mol, smiles=smiles)


@dataclass
class Batch:
    """Collated batch of ``MolData`` (like PyG's ``Batch.from_data_list``).

    edge_index/rev_index offset per-graph. atom_batch/edge_batch: graph
    index each atom/edge belongs to.
    """

    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    rev_index: np.ndarray
    atom_batch: np.ndarray
    edge_batch: np.ndarray
    num_graphs: int


def batch_from_data_list(data_list: Sequence[MolData]) -> Batch:
    xs, edge_indices, edge_attrs, rev_indices, atom_batches = [], [], [], [], []
    atom_offset = 0
    edge_offset = 0
    for i, d in enumerate(data_list):
        n_atoms = d.x.shape[0]
        xs.append(d.x)
        edge_indices.append(d.edge_index + atom_offset)
        edge_attrs.append(d.edge_attr)
        rev_indices.append(d.rev_index + edge_offset)
        atom_batches.append(np.full((n_atoms,), i, dtype=np.int32))
        atom_offset += n_atoms
        edge_offset += d.edge_index.shape[1]

    x = np.concatenate(xs, axis=0) if xs else np.zeros((0, atom_feature_dim()), dtype=np.float32)
    edge_index = np.concatenate(edge_indices, axis=1) if edge_indices else np.zeros((2, 0), dtype=np.int32)
    edge_attr = np.concatenate(edge_attrs, axis=0) if edge_attrs else np.zeros((0, bond_feature_dim()), dtype=np.float32)
    rev_index = np.concatenate(rev_indices, axis=0) if rev_indices else np.zeros((0,), dtype=np.int32)
    atom_batch = np.concatenate(atom_batches, axis=0) if atom_batches else np.zeros((0,), dtype=np.int32)
    edge_batch = atom_batch[edge_index[0]] if edge_index.shape[1] > 0 else np.zeros((0,), dtype=np.int32)

    return Batch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        rev_index=rev_index,
        atom_batch=atom_batch,
        edge_batch=edge_batch,
        num_graphs=len(data_list),
    )


def smiles_list_to_batch(smiles_list: list[str]) -> Batch:
    return batch_from_data_list([smiles_to_data(s) for s in smiles_list])


def bond_init_message(x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> np.ndarray:
    """Directed-bond init message: concat(source atom features, bond features)."""
    return np.concatenate([x[edge_index[0]], edge_attr], axis=-1)
