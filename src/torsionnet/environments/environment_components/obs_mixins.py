import numpy as np
import torch

from rdkit import Chem

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance, NormalizeScale, Center, NormalizeRotation
from torsionnet.environments.environment_components import molecule_features

class GraphObsMixin:
    def _obs(self):
        mol = Chem.rdmolops.RemoveHs(self.mol)
        conf = mol.GetConformer()
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        node_features = [molecule_features.atom_coords(atom, conf) for atom in atoms]
        edge_indices = molecule_features.get_bond_pairs(mol)
        edge_attributes = [molecule_features.bond_type(bond) for bond in bonds] * 2

        data = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_indices, dtype=torch.long),
                    edge_attr=torch.tensor(edge_attributes,dtype=torch.float),
                    pos=torch.Tensor(conf.GetPositions())
                )

        data = Center()(data)
        data = NormalizeRotation()(data)
        data.x = data.pos
        data = Batch.from_data_list([data])
        return data, self.nonring

class AtomTypeGraphObsMixin:
    def _obs(self):
        mol = Chem.rdmolops.RemoveHs(self.mol)
        conf = mol.GetConformer()
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        node_features = [molecule_features.atom_type_CO(atom) for atom in atoms]
        edge_indices = molecule_features.get_bond_pairs(mol)
        edge_attributes = [molecule_features.bond_type(bond) for bond in bonds] * 2

        data = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_indices, dtype=torch.long),
                    edge_attr=torch.tensor(edge_attributes,dtype=torch.float),
                    pos=torch.Tensor(conf.GetPositions())
                )

        data = Center()(data)
        data = NormalizeRotation()(data)
        data = Batch.from_data_list([data])
        return data, self.nonring


class AtomCoordsTypeGraphObsMixin:
    def _obs(self):
        mol = Chem.rdmolops.RemoveHs(self.mol)
        conf = mol.GetConformer()
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        node_features = [molecule_features.atom_type_CO(atom) + molecule_features.atom_coords(atom, conf) for atom in atoms]
        edge_indices = molecule_features.get_bond_pairs(mol)
        edge_attributes = [molecule_features.bond_type(bond) for bond in bonds] * 2


        data = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_indices, dtype=torch.long),
                    edge_attr=torch.tensor(edge_attributes,dtype=torch.float),
                    pos=torch.Tensor(conf.GetPositions())
                )

        data = Center()(data)
        data = NormalizeRotation()(data)
        data.x[:,-3:] = data.pos
        data = Batch.from_data_list([data])
        return data, self.nonring