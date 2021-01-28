import numpy as np
import torch

from rdkit import Chem

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance, NormalizeScale, Center, NormalizeRotation

from .conformer_env import ConformerEnv, confgen

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

class SkeletonPointsObsMixin(ConformerEnv):
    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeletonpoints(self.molecule)])
        return data, self.nonring

class XorgateSkeletonPointsObsMixin(ConformerEnv):

    bonds = False

    def _get_obs(self):
        mol = self.molecule

        if not self.bonds:
            self.bonds = True
            self.bbatomidx = torch.tensor([
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [10, 11, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
                [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 38, 39],
                [24, 25, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
            ], dtype=torch.long)
            self.bbatoms = [
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                {10, 11, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43},
                {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 38, 39},
                {24, 25, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57}
            ]

            bonds = mol.GetBonds()
            self.edge_index = get_bond_pair(mol)

            self.edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=True) for bond in bonds]
            for bond in bonds:
                self.edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=True))

            self.bb_edge_index = [[[], []], [[], []], [[], []], [[], []]]
            self.bb_edge_attr = [[], [], [], []]

            for i in range(len(self.edge_index[0])):
                for j in range(4):
                    if self.edge_index[0][i] in self.bbatoms[j] and self.edge_index[1][i] in self.bbatoms[j]:
                        self.bb_edge_index[j][0].append(self.edge_index[0][i])
                        self.bb_edge_index[j][1].append(self.edge_index[1][i])
                        self.bb_edge_attr[j].append(self.edge_attr[i])

            self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
            self.edge_attr = torch.tensor(self.edge_attr, dtype=torch.float)

        atoms = mol.GetAtoms()
        conf = mol.GetConformer(id=-1)
        node_f = [atom_features(atom, conf) for atom in atoms]
        node_f = torch.tensor(node_f, dtype=torch.float)
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

        data = Data(
                x=node_f,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                pos=pos)
            
        bbdata = []
        for i in range(4):
            bbdata.append(
                Data(
                    x=torch.index_select(node_f, 0, self.bbatomidx[i]),
                    edge_index=torch.tensor(self.bb_edge_index[i], dtype=torch.long),
                    edge_attr=torch.tensor(self.bb_edge_attr[i], dtype=torch.float),
                    pos=torch.index_select(pos, 0, self.bbatomidx[i])
                )
            )
            bbdata[i] = Center()(bbdata[i])
            bbdata[i] = NormalizeRotation()(bbdata[i])
            bbdata[i].x[:, -3:] = bbdata[i].pos
            bbdata[i] = Batch.from_data_list([bbdata[i]])

        data = Center()(data)
        data = NormalizeRotation()(data)
        data.x[:,-3:] = data.pos
        data = Batch.from_data_list([data])
        return data, bbdata, self.nonring