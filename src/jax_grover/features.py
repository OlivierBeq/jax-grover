"""Atom/bond featurization, matching GROVER's exact feature layout
(pretrained checkpoints depend on it). Pure numpy/rdkit, no JAX.
"""


import numpy as np
from rdkit import Chem

MAX_ATOMIC_NUM = 100

ATOM_FEATURES = {
    "atomic_num": list(range(MAX_ATOMIC_NUM)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def atom_feature_dim() -> int:
    """Dimensionality of a single atom's feature vector (151)."""
    return ATOM_FDIM + 18


def bond_feature_dim() -> int:
    """Dimensionality of a single bond's feature vector (14)."""
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: list[int]) -> list[int]:
    """One-hot encoding of length ``len(choices) + 1``.

    Quirk (must be preserved for checkpoint compatibility): if ``choices``
    contains a negative value (only ``formal_charge`` does), ``value`` is
    used directly as the index instead of ``choices.index(value)``.
    """
    encoding = [0] * (len(choices) + 1)
    if min(choices) < 0:
        index = value
    else:
        index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


class MolFeatureContext:
    """Precomputed whole-molecule SMARTS matches, reused across all atoms of a mol."""

    _HYDROGEN_DONOR = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
    _HYDROGEN_ACCEPTOR = Chem.MolFromSmarts(
        "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
        "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"
    )
    _ACIDIC = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
    _BASIC = Chem.MolFromSmarts(
        "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
        "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]"
    )

    def __init__(self, mol: Chem.rdchem.Mol):
        self.hydrogen_donor_match = sum(mol.GetSubstructMatches(self._HYDROGEN_DONOR), ())
        self.hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self._HYDROGEN_ACCEPTOR), ())
        self.acidic_match = sum(mol.GetSubstructMatches(self._ACIDIC), ())
        self.basic_match = sum(mol.GetSubstructMatches(self._BASIC), ())
        self.ring_info = mol.GetRingInfo()


def atom_features(atom: Chem.rdchem.Atom, context: MolFeatureContext) -> list[bool | int | float]:
    """151-dim feature vector for a single atom."""
    features = (
        onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES["num_Hs"])
        + onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES["hybridization"])
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )
    atom_idx = atom.GetIdx()
    features = (
        features
        + onek_encoding_unk(atom.GetValence(Chem.ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6])
        + [atom_idx in context.hydrogen_acceptor_match]
        + [atom_idx in context.hydrogen_donor_match]
        + [atom_idx in context.acidic_match]
        + [atom_idx in context.basic_match]
        + [
            context.ring_info.IsAtomInRingOfSize(atom_idx, 3),
            context.ring_info.IsAtomInRingOfSize(atom_idx, 4),
            context.ring_info.IsAtomInRingOfSize(atom_idx, 5),
            context.ring_info.IsAtomInRingOfSize(atom_idx, 6),
            context.ring_info.IsAtomInRingOfSize(atom_idx, 7),
            context.ring_info.IsAtomInRingOfSize(atom_idx, 8),
        ]
    )
    return features


def bond_features(bond: Chem.rdchem.Bond) -> list[bool | int | float]:
    """14-dim feature vector for a single bond (bond-only, no source-atom concatenation)."""
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0),
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def atom_features_array(mol: Chem.rdchem.Mol) -> np.ndarray:
    """(n_atoms, 151) float32 atom feature matrix for a whole molecule."""
    context = MolFeatureContext(mol)
    n_atoms = mol.GetNumAtoms()
    x = np.zeros((n_atoms, atom_feature_dim()), dtype=np.float32)
    for atom in mol.GetAtoms():
        x[atom.GetIdx()] = atom_features(atom, context)
    return x
