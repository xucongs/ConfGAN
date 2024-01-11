import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from rdkit.Chem import BRICS


class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()

    def is_aromatic(self, atom):
        return atom.GetIsAromatic()

    def chiral_types(self, atom):
        return atom.GetChiralTag()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets, hop_num):
        super().__init__(allowable_sets)
        self.dim += (hop_num - 1 + 2)

    def encode(self, bond, hop, hop_num, is_aromatic1=False, is_aromatic2=False):
        output = np.zeros((self.dim,))
        if bond is None and hop:
            output[-2] = int(is_aromatic1)
            output[-1] = int(is_aromatic2)
            output[hop - hop_num - 3] = 1
            return output
        output = super().encode(bond)
        output[-2] = int(is_aromatic1)
        output[-1] = int(is_aromatic2)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()


def add_hop_pair(atom_features, pair_indices, hop_num=3):
    diag = np.diag([1 for _ in range(atom_features.shape[0])])
    pair_indices_mat = np.zeros([atom_features.shape[0], atom_features.shape[0]]) + diag
    pair_indices_mat[pair_indices[:, 0], pair_indices[:, 1]] = 1
    pair_indices_list = [diag, pair_indices_mat]

    for i in range(2, hop_num + 1):
        x = pair_indices_list[i - 1] @ pair_indices_list[1]
        pair_indices_list.append(np.where(x > 0, np.ones_like(x), np.zeros_like(x)))
    hop_pair_indices = np.zeros_like(pair_indices_mat)

    for i in range(1, hop_num + 1):
        hop_pair_indices += (pair_indices_list[i] - pair_indices_list[i - 1]) * i
    return hop_pair_indices


def graph_from_mol(mol, hop_num=0, is_train=True):
    atom_featurizer = AtomFeaturizer(
        allowable_sets={
            "symbol": {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "I"},
            "n_valence": {0, 1, 2, 3, 4, 5, 6},
            "n_hydrogens": {0, 1, 2, 3, 4},
            "hybridization": {"s", "sp", "sp2", "sp3"},
            "is_aromatic": {False, True},
            "chiral_types": {
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            }
        }
    )

    bond_featurizer = BondFeaturizer(
        allowable_sets={
            "bond_type": ["single", "double", "triple", "aromatic"],
        }, hop_num=hop_num
    )

    atom_features = []
    bond_features = []
    pair_idx = []
    pair_indices= []
    pair_distance = []
    hops = []
    u_parm = []

    for atom in mol.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))
    for bond in mol.GetBonds():
        pair_idx.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        pair_idx.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    pair_idx = np.array(pair_idx)
    atom_features = np.array(atom_features)

    atoms_mofit, pair_motif, break_bonds = brics_decomp(mol)
    motif_bonds = [bond_featurizer.encode(mol.GetBondBetweenAtoms(int(i), int(j)), 1, 1) for i, j in break_bonds]

    if hop_num:
        pair_indices_mat = add_hop_pair(atom_features, pair_idx)

    for hop in range(1, hop_num + 1):

        hop_indices = np.where(pair_indices_mat == hop)

        pair_all = [[i, j] for i, j in zip(hop_indices[0], hop_indices[1])]
        for i, j in pair_all:
            if [i, j] not in pair_indices and [j, i] not in pair_indices:
                pair_indices.append([i, j])
                bond_features.append(bond_featurizer.encode(mol.GetBondBetweenAtoms(int(i), int(j)), hop, hop_num,
                                                            mol.GetAtoms()[int(i)].GetIsAromatic(),
                                                            mol.GetAtoms()[int(j)].GetIsAromatic()))
                if is_train:
                    pair_distance.append(AllChem.Get3DDistanceMatrix(mol)[int(i), int(j)])
                    hops.append(hop)
                if hop == 1:
                    bondstretchparam = AllChem.GetUFFBondStretchParams(mol, int(i), int(j)) + (0,)
                    u_parm.append(bondstretchparam)
                else:
                    getvdwparam = AllChem.GetUFFVdWParams(mol, int(i), int(j)) + (1,)
                    u_parm.append(getvdwparam)
    return atom_features, bond_features, pair_indices, atoms_mofit, pair_motif, motif_bonds, pair_distance, hops, u_parm


def graphs_from_mols(mols, hop_num=3, is_train=True):
    # Initialize graphs
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []
    atoms_mofit_list = []
    pair_motif_list = []
    motif_bonds_list = []
    distance_list = []
    hop_indices_list = []
    u_parm_list = []
    for mol in mols:

        try:
            atom_features, bond_features, pair_indices, atoms_mofit, pair_motif, motif_bonds, pair_distance, hop_indices, u_parm = graph_from_mol(mol, hop_num, is_train=is_train)
            atom_features_list.append(atom_features)
            bond_features_list.append(bond_features)
            pair_indices_list.append(pair_indices)
            atoms_mofit_list.append(atoms_mofit)
            pair_motif_list.append(pair_motif)
            motif_bonds_list.append(motif_bonds)
            distance_list.append(pair_distance)
            hop_indices_list.append(hop_indices)
            u_parm_list.append(u_parm)
        except Exception as e:
            print(e)
            print(AllChem.MolToSmiles(mol))
            continue

    data_dict = {
        'atom_features': tf.ragged.constant(atom_features_list, dtype=tf.float32),
        'bond_features': tf.ragged.constant(bond_features_list, dtype=tf.float32),
        'pair_indices': tf.ragged.constant(pair_indices_list, dtype=tf.int32),
        'atoms_mofit': tf.ragged.constant(atoms_mofit_list, dtype=tf.int32),
        'pair_motif_indices': tf.ragged.constant(pair_motif_list, dtype=tf.int32),
        'motif_bonds': tf.ragged.constant(motif_bonds_list, dtype=tf.float32),
        'u_parm': tf.ragged.constant(u_parm_list, dtype=tf.float32),
        'hop_indices': tf.ragged.constant(hop_indices_list, dtype=tf.int16),
        'distance': tf.ragged.constant(distance_list, dtype=tf.float32),
    }
    
    return data_dict 



# molecular motif
def brics_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))

    for bond in res:
        if [bond[0][0], bond[0][1]] in cliques:
            cliques.remove([bond[0][0], bond[0][1]])
        else:
            cliques.remove([bond[0][1], bond[0][0]])
        cliques.append([bond[0][0]])
        cliques.append([bond[0][1]])
    cliques.sort()
    
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)

    # select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])
    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]

    cliques = [c for c in cliques if len(c) > 0]

    # edges
    edges = []

    for bond in res:
        for c in range(len(cliques)):
            if bond[0][0] in cliques[c]:
                c1 = c
            if bond[0][1] in cliques[c]:
                c2 = c
        edges.append([c1, c2])
    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append([c1, c2])
    break_bonds = [list(i[0]) for i in res] + breaks
    if edges == []:
        edges = [[0, 0]]
        break_bonds = [[0, 0]]
    return cliques, edges, break_bonds
