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