import numpy as np
import torch

from rdkit import Chem

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance, NormalizeScale, Center, NormalizeRotation

from .conformer_env import ConformerEnv

def bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = []
    bond_feats = bond_feats + [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
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

class SkeletonPointsObsMixin(ConformerEnv):
    def _get_obs(self):
        
        mol = self.molecule
        mol = Chem.rdmolops.RemoveHs(mol)
        conf = mol.GetConformer(id=-1)
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        node_f = [atom_features(atom, conf) for atom in atoms]
        edge_index = get_bond_pair(mol)
        edge_attr = [bond_features(bond) for bond in bonds]
        for bond in bonds:
            edge_attr.append(bond_features(bond))

        data = Data(
                    x=torch.tensor(node_f, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                    pos=torch.Tensor(conf.GetPositions())
                )

        data = Center()(data)
        data = NormalizeRotation()(data)
        data.x[:,-3:] = data.pos
        data = Batch.from_data_list([data])
        return data, self.nonring

class XorgateSkeletonPointsObsMixin(ConformerEnv):
    bonds = False

    def _get_obs(self):
        mol = self.molecule

        if not self.bonds:
            self.bonds = True
            self.bbatomidx = torch.tensor([
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 39, 38, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                [40, 32, 31, 37, 38, 39, 41, 42, 43, 33, 11, 10, 30, 34, 35, 36],
            ], dtype=torch.long)

            bonds = mol.GetBonds()
            self.edge_index = get_bond_pair(mol)

            self.edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=True) for bond in bonds]
            for bond in bonds:
                self.edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=True))

            self.bb_edge_index = [[[], []], [[], []], [[], []]]
            self.bb_edge_attr = [[], [], []]

            for i in range(len(self.edge_index[0])):
                if self.edge_index[0][i] <= 15 and self.edge_index[1][i] <= 15:
                    for j in range(3):
                        self.bb_edge_index[j][0].append(self.edge_index[0][i])
                        self.bb_edge_index[j][1].append(self.edge_index[1][i])
                        self.bb_edge_attr[j].append(self.edge_attr[i])

            self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
            self.edge_attr = torch.tensor(self.edge_attr, dtype=torch.float)
            self.bb_edge_index = torch.tensor(self.bb_edge_index, dtype=torch.long)
            self.bb_edge_attr = torch.tensor(self.bb_edge_attr, dtype=torch.float)

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
        for i in range(3):
            bbdata.append(
                Data(
                    x=torch.index_select(node_f, 0, self.bbatomidx[i]),
                    edge_index=self.bb_edge_index[i],
                    edge_attr=self.bb_edge_attr[i],
                    pos=torch.index_select(pos, 0, self.bbatomidx[i])
                )
            )
            bbdata[i] = Center()(bbdata[i])
            bbdata[i] = NormalizeRotation()(bbdata[i])
            bbdata[i].x[:, -3:] = bbdata[i].pos
            bbdata[i] = Batch.from_data_list([bbdata[i]])

        data = Center()(data)
        data = NormalizeRotation()(data)
        data.x[:, -3:] = data.pos
        data = Batch.from_data_list([data])
        return data, bbdata, self.nonring