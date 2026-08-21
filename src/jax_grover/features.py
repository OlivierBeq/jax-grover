"""Atom/bond featurization, matching GROVER's exact feature layout
(pretrained checkpoints depend on it). Pure numpy/rdkit, no JAX.

The `_array` variants vectorize the one-hot/boolean encoding with numpy
fancy indexing instead of per-atom/bond Python list building.
"""

import itertools

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


def _lookup_table(choices: list, table_size: int) -> np.ndarray:
    """``table[v] == choices.index(v)``, else the unknown bucket ``len(choices)``."""
    table = np.full(table_size, len(choices), dtype=np.int64)
    for i, v in enumerate(choices):
        v = int(v)
        if 0 <= v < table_size:
            table[v] = i
    return table


def _make_onehot_indexer(choices: list, table_size: int):
    """Vectorized ``onek_encoding_unk`` index lookup."""
    table = _lookup_table(choices, table_size)
    unknown_index = len(choices)

    def indexer(values: np.ndarray) -> np.ndarray:
        in_range = (values >= 0) & (values < table_size)
        safe_values = np.where(in_range, values, 0)
        return np.where(in_range, table[safe_values], unknown_index)

    return indexer, unknown_index


_ATOMIC_NUM_INDEXER, _ = _make_onehot_indexer(ATOM_FEATURES["atomic_num"], 100)
_DEGREE_INDEXER, _ = _make_onehot_indexer(ATOM_FEATURES["degree"], 32)
_CHIRAL_TAG_INDEXER, _ = _make_onehot_indexer(ATOM_FEATURES["chiral_tag"], 32)
_NUM_HS_INDEXER, _ = _make_onehot_indexer(ATOM_FEATURES["num_Hs"], 32)
_HYBRIDIZATION_INDEXER, _ = _make_onehot_indexer([int(h) for h in ATOM_FEATURES["hybridization"]], 32)
_VALENCE_CHOICES = [0, 1, 2, 3, 4, 5, 6]
_VALENCE_INDEXER, _ = _make_onehot_indexer(_VALENCE_CHOICES, 32)
_STEREO_CHOICES = list(range(6))
_STEREO_INDEXER, _ = _make_onehot_indexer(_STEREO_CHOICES, 32)

_FORMAL_CHARGE_WIDTH = len(ATOM_FEATURES["formal_charge"]) + 1  # 6

_BOND_TYPES = (
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
)


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
        # frozenset for O(1) membership below.
        self.hydrogen_donor_match = frozenset(itertools.chain.from_iterable(mol.GetSubstructMatches(self._HYDROGEN_DONOR)))
        self.hydrogen_acceptor_match = frozenset(
            itertools.chain.from_iterable(mol.GetSubstructMatches(self._HYDROGEN_ACCEPTOR))
        )
        self.acidic_match = frozenset(itertools.chain.from_iterable(mol.GetSubstructMatches(self._ACIDIC)))
        self.basic_match = frozenset(itertools.chain.from_iterable(mol.GetSubstructMatches(self._BASIC)))
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


def bond_features_array(bonds: list) -> np.ndarray:
    """(n_bonds, 14) float32 bond feature matrix for a list of real (non-None) RDKit bonds."""
    n = len(bonds)
    x = np.zeros((n, bond_feature_dim()), dtype=np.float32)
    if n == 0:
        return x

    bond_type = np.empty(n, dtype=np.int64)
    conjugated = np.empty(n, dtype=np.float32)
    in_ring = np.empty(n, dtype=np.float32)
    stereo = np.empty(n, dtype=np.int64)
    for i, bond in enumerate(bonds):
        bond_type[i] = int(bond.GetBondType())
        conjugated[i] = 1.0 if bond.GetIsConjugated() else 0.0
        in_ring[i] = 1.0 if bond.IsInRing() else 0.0
        stereo[i] = int(bond.GetStereo())

    x[:, 1] = bond_type == int(_BOND_TYPES[0])
    x[:, 2] = bond_type == int(_BOND_TYPES[1])
    x[:, 3] = bond_type == int(_BOND_TYPES[2])
    x[:, 4] = bond_type == int(_BOND_TYPES[3])
    x[:, 5] = conjugated
    x[:, 6] = in_ring

    row = np.arange(n)
    x[row, 7 + _STEREO_INDEXER(stereo)] = 1.0
    return x


def atom_features_array(mol: Chem.rdchem.Mol) -> np.ndarray:
    """(n_atoms, 151) float32 atom feature matrix for a whole molecule."""
    context = MolFeatureContext(mol)
    n_atoms = mol.GetNumAtoms()
    x = np.zeros((n_atoms, atom_feature_dim()), dtype=np.float32)
    if n_atoms == 0:
        return x

    atomic_num = np.empty(n_atoms, dtype=np.int64)
    degree = np.empty(n_atoms, dtype=np.int64)
    formal_charge = np.empty(n_atoms, dtype=np.int64)
    chiral_tag = np.empty(n_atoms, dtype=np.int64)
    num_hs = np.empty(n_atoms, dtype=np.int64)
    hybridization = np.empty(n_atoms, dtype=np.int64)
    aromatic = np.empty(n_atoms, dtype=np.float32)
    mass = np.empty(n_atoms, dtype=np.float32)
    valence = np.empty(n_atoms, dtype=np.int64)

    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        atomic_num[i] = atom.GetAtomicNum() - 1
        degree[i] = atom.GetTotalDegree()
        formal_charge[i] = atom.GetFormalCharge()
        chiral_tag[i] = int(atom.GetChiralTag())
        num_hs[i] = int(atom.GetTotalNumHs())
        hybridization[i] = int(atom.GetHybridization())
        aromatic[i] = 1.0 if atom.GetIsAromatic() else 0.0
        mass[i] = atom.GetMass() * 0.01
        valence[i] = atom.GetValence(Chem.ValenceType.IMPLICIT)

    row = np.arange(n_atoms)
    col = 0

    def write_onehot(values: np.ndarray, indexer, width: int) -> None:
        nonlocal col
        x[row, col + indexer(values)] = 1.0
        col += width

    write_onehot(atomic_num, _ATOMIC_NUM_INDEXER, len(ATOM_FEATURES["atomic_num"]) + 1)
    write_onehot(degree, _DEGREE_INDEXER, len(ATOM_FEATURES["degree"]) + 1)

    # formal_charge quirk: raw value as index, incl. negative wraparound (mod).
    x[row, col + np.mod(formal_charge, _FORMAL_CHARGE_WIDTH)] = 1.0
    col += _FORMAL_CHARGE_WIDTH

    write_onehot(chiral_tag, _CHIRAL_TAG_INDEXER, len(ATOM_FEATURES["chiral_tag"]) + 1)
    write_onehot(num_hs, _NUM_HS_INDEXER, len(ATOM_FEATURES["num_Hs"]) + 1)
    write_onehot(hybridization, _HYBRIDIZATION_INDEXER, len(ATOM_FEATURES["hybridization"]) + 1)

    x[:, col] = aromatic
    col += 1
    x[:, col] = mass
    col += 1

    write_onehot(valence, _VALENCE_INDEXER, len(_VALENCE_CHOICES) + 1)

    def write_bool_col(match: frozenset) -> None:
        nonlocal col
        if match:
            idx = np.fromiter(match, dtype=np.int64, count=len(match))
            x[idx, col] = 1.0
        col += 1

    write_bool_col(context.hydrogen_acceptor_match)
    write_bool_col(context.hydrogen_donor_match)
    write_bool_col(context.acidic_match)
    write_bool_col(context.basic_match)

    for ring in context.ring_info.AtomRings():
        size = len(ring)
        if 3 <= size <= 8:
            idx = np.fromiter(ring, dtype=np.int64, count=size)
            x[idx, col + (size - 3)] = 1.0
    col += 6

    return x
