"""
Observation_mixins
==================

Pre-built observation handlers.
"""
import numpy as np
import torch

from rdkit import Chem

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance, NormalizeScale, Center, NormalizeRotation
from conformer_rl.environments.environment_components import molecule_features

from typing import List, Tuple

class GraphObsMixin:
    """Represents molecule as a PyTorch Geometric graph where each node contains information about the atom's three-dimensional coordinates.
    """
    def _obs(self) -> Tuple[Batch, List[List[int]]]:
        """
        returns
        -------
        Tuple[Batch, List[List[int]]
            The Batch object contains the Pytorch Geometric graph representing the molecule. The list of lists of integers
            is a list of all the torsions of the molecule, where each torsion is represented by a list of four integers, where the integers
            are the indices of the four atoms making up the torsion.
        """

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
    """Represents molecule as a PyTorch Geometric graph where each node contains information about the atom's element.
    """
    def _obs(self) -> Tuple[Batch, List[List[int]]]:
        """
        returns
        -------
        Tuple[Batch, List[List[int]]
            The Batch object contains the Pytorch Geometric graph representing the molecule. The list of lists of integers
            is a list of all the torsions of the molecule, where each torsion is represented by a list of four integers, where the integers
            are the indices of the four atoms making up the torsion.
        """
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
    """Represents molecule as a PyTorch Geometric graph where each node contains information about the atom's element and three-dimensional coordinates.
    """
    def _obs(self) -> Tuple[Batch, List[List[int]]]:
        """
        returns
        -------
        Tuple[Batch, List[List[int]]
            The Batch object contains the Pytorch Geometric graph representing the molecule. The list of lists of integers
            is a list of all the torsions of the molecule, where each torsion is represented by a list of four integers, where the integers
            are the indices of the four atoms making up the torsion.
        """
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