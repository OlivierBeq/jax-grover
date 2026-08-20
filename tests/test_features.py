"""Feature-extraction regression tests, pinned against a frozen golden
fixture (tests/fixtures/features_golden.npz)."""

from pathlib import Path

import numpy as np
from rdkit import Chem

from jax_grover.features import atom_feature_dim, atom_features_array, bond_feature_dim, bond_features

FIXTURE = Path(__file__).parent / "fixtures" / "features_golden.npz"


def test_feature_dims():
    assert atom_feature_dim() == 151
    assert bond_feature_dim() == 14


def test_atom_features_match_golden_fixture():
    data = np.load(FIXTURE)
    smiles_list = data["smiles"]
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smiles))
        ours = atom_features_array(mol)
        golden = data[f"atom_features_{i}"]
        np.testing.assert_array_equal(ours, golden)


def test_bond_features_match_golden_fixture():
    data = np.load(FIXTURE)
    smiles_list = data["smiles"]
    golden = data["bond_features"]

    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(str(smiles))
        for bond in mol.GetBonds():
            rows.append(np.array(bond_features(bond), dtype=np.float32))
    ours = np.stack(rows) if rows else np.zeros((0, 14), dtype=np.float32)

    np.testing.assert_array_equal(ours, golden)


def test_onek_encoding_unk_negative_choice_uses_raw_value_as_index():
    """formal_charge's choices contain a negative value, so it indexes with
    the raw value instead of ``choices.index(value)``."""
    mol = Chem.MolFromSmiles("C")
    atom = mol.GetAtomWithIdx(0)
    assert atom.GetFormalCharge() == 0

    features = atom_features_array(mol)[0]
    # atomic_num one-hot: 100 choices + 1 unknown slot = width 101, offset 0.
    # degree one-hot: 6 choices + 1 = width 7, offset 101.
    # formal_charge one-hot: 5 choices + 1 = width 6, offset 108.
    formal_charge_slice = features[108:114]
    expected = np.zeros(6, dtype=np.float32)
    expected[0] = 1.0  # raw value 0 used directly as index
    np.testing.assert_array_equal(formal_charge_slice, expected)
