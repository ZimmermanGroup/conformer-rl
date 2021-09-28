from conformer_rl.environments.environment_components.molecule_features import bond_type, get_bond_pairs, atom_coords, atom_type_CO
import pytest
from rdkit import Chem


def test_bond_type():
    mol = Chem.MolFromSmiles('C(=O)(C(=O)O)O')
    bonds = list(mol.GetBonds())
    assert bond_type(bonds[0]) == [0, 1, 0, 0, 1, 0]
    assert bond_type(bonds[1]) == [1, 0, 0, 0, 1, 0]
    mol = Chem.MolFromSmiles('C#C')
    bonds = list(mol.GetBonds())
    assert bond_type(bonds[0]) == [0, 0, 1, 0, 0, 0]
    mol = Chem.MolFromSmiles('C1=CC=CC=C1')
    bonds = list(mol.GetBonds())
    assert bond_type(bonds[0]) == [0, 0, 0, 1, 1, 1]

def test_get_bond_pairs():
    mol = Chem.MolFromSmiles("CC(CCC)CCCC(CCCC)CC")
    assert get_bond_pairs(mol) == [
        [0, 1, 1, 2, 2, 3, 3, 4, 1, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 8, 13, 13, 14], 
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 1, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11, 10, 12, 11, 13, 8, 14, 13]
    ]

def test_atom_coords(mocker):
    pos = mocker.Mock()
    pos.x = 'x'
    pos.y = 'y'
    pos.z = 'z'

    conf = mocker.Mock()
    conf.GetAtomPosition.return_value = pos

    atom = mocker.Mock()
    atom.GetIdx.return_value = 'idx'

    fts = atom_coords(atom, conf)
    conf.GetAtomPosition.assert_called_once_with('idx')
    assert fts == ['x', 'y', 'z']

def test_atom_type_CO(mocker):
    carbon = mocker.Mock()
    carbon.GetSymbol.return_value = 'C'
    oxygen = mocker.Mock()
    oxygen.GetSymbol.return_value = 'O'

    assert atom_type_CO(carbon) == [1, 0]
    assert atom_type_CO(oxygen) == [0, 1]

