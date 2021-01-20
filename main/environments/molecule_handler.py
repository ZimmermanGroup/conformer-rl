import numpy as np
import torch

from rdkit import Chem, DataStructs, RDConfig, rdBase
from rdkit import rdBase
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.Chem import Draw,PyMol,rdFMCS

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance, NormalizeScale, Center, NormalizeRotation

def bond_features(bond, use_chirality=False, use_basic_feats=True, null_feature=False):
    bt = bond.GetBondType()
    bond_feats = []
    if use_basic_feats:
        bond_feats = bond_feats + [
            bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    if null_feature:
        bond_feats += [0.0]
    return np.array(bond_feats)

def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def atom_features(atom, conf):

    anum = atom.GetSymbol()
    atom_feats = []

    atom_feats = atom_feats + [
        anum == 'C', anum == 'O',
    ]

    p = conf.GetAtomPosition(atom.GetIdx())
    fts = atom_feats + [p.x, p.y, p.z]
    return np.array(fts)


def atom_features_simple(atom, conf):
    p = conf.GetAtomPosition(atom.GetIdx())
    return np.array([p.x, p.y, p.z])

def mol2vecsimple(mol):
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [atom_features_simple(atom, conf) for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False) for bond in bonds]
    for bond in bonds:
        edge_attr.append(bond_features(bond))
    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )
    data = Distance()(data)
    return data

def mol2vecstupidsimple(mol):
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [[] for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=False) for bond in bonds]
    for bond in bonds:
        edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=False))
    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    data = NormalizeScale()(data)
    data = Distance(norm=False)(data)
    data.x = data.pos

    e = data.edge_attr
    new_e = -1 + ((e - e.min())*2)/(e.max() - e.min())
    data.edge_attr = new_e

    return data

def mol2vecskeleton(mol):
    mol = Chem.rdmolops.RemoveHs(mol)
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f = [[] for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=False) for bond in bonds]

    for bond in bonds:
        edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=False))

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    data = NormalizeScale()(data)
    data = Distance(norm=False)(data)
    data.x = data.pos

    return data

def mol2vecskeleton_features(mol):
    mol = Chem.rdmolops.RemoveHs(mol)
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [[] for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=True) for bond in bonds]
    for bond in bonds:
        edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=True))

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    data = NormalizeScale()(data)
    data = Distance(norm=False)(data)
    data.x = data.pos

    return data

def mol2vecdense(mol):
    mol = Chem.rdmolops.RemoveHs(mol)
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    n = len(atoms)

    edge_index = []
    edge_attr = []


    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edge_index.append([i, j])
            edge_attr.append(adj[i][j])


    node_f= [[] for atom in atoms]

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).T,
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    data = NormalizeScale()(data)
    data = Distance(norm=False)(data)
    data.x = data.pos

    return data

def mol2vecbasic(mol):
    mol = Chem.rdmolops.RemoveHs(mol)
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [[] for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=False) for bond in bonds]
    for bond in bonds:
        edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=False))

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    data = NormalizeScale()(data)
    data = Distance(norm=False)(data)
    data.x = data.pos

    e = data.edge_attr
    new_e = -1 + ((e - e.min())*2)/(e.max() - e.min())
    data.edge_attr = new_e

    return data

def mol2vecskeletonpoints(mol):
    mol = Chem.rdmolops.RemoveHs(mol)
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f = [atom_features(atom, conf) for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=True) for bond in bonds]
    for bond in bonds:
        edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=True))

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    data = Center()(data)
    data = NormalizeRotation()(data)
    data.x[:,-3:] = data.pos
    
    assert (data.x == data.x).all()
    assert (data.edge_attr == data.edge_attr).all()
    assert (data.edge_index == data.edge_index).all()

    return data

def mol2vecskeletonpoints_test(mol):
    mol = Chem.rdmolops.RemoveHs(mol)
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f = [atom_features(atom, conf) for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=True) for bond in bonds]
    for bond in bonds:
        edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=True))

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    return data

def mol2vecskeletonpointswithdistance(mol):
    mol = Chem.rdmolops.RemoveHs(mol)
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f = [atom_features(atom, conf) for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=True) for bond in bonds]
    for bond in bonds:
        edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=True))

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    data = Center()(data)
    data = NormalizeRotation()(data)
    data = Distance(norm=False)(data)
    data.x[:,-3:] = data.pos

    return data


def mol2points(mol):
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [atom_features_simple(atom, conf) for atom in atoms]

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
            )
    data = Distance()(data)
    return data