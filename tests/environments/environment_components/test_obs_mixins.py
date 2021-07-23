from conformer_rl.environments.environment_components.obs_mixins import GraphObsMixin, AtomTypeGraphObsMixin, AtomCoordsTypeGraphObsMixin
import torch

def atype(atom):
    d = {
        'a1': [1, 0],
        'a2': [0, 1],
        'a3': [1, 0]
    }
    return d[atom]

def acoords(atom, conf):
    d = {
        'a1': [1, 2, 3],
        'a2': [4, 5, 6],
        'a3': [7, 8, 9]
    }
    return d[atom]

def btype(bond):
    d = {
        'b1': [1, 0, 1, 1, 0, 1],
        'b2': [0, 1, 1, 0, 1, 1]
    }
    return d[bond]

def test_GraphObsMixin(mocker):
    mol = mocker.Mock()
    mol.GetConformer.return_value = mol.conf
    mol.conf.return_value = 'conf'
    mol.conf.GetPositions.return_value = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    mol.GetAtoms.return_value = ['a1', 'a2', 'a3']
    mol.GetBonds.return_value = ['b1', 'b2']

    removeHs = mocker.patch('conformer_rl.environments.environment_components.obs_mixins.Chem.rdmolops.RemoveHs')
    removeHs.return_value = mol
    bond_type = mocker.patch('conformer_rl.environments.environment_components.obs_mixins.molecule_features.bond_type')
    bond_type.side_effect = btype
    atom_coords = mocker.patch('conformer_rl.environments.environment_components.obs_mixins.molecule_features.atom_coords')
    atom_coords.side_effect = acoords
    bond_pairs = mocker.patch('conformer_rl.environments.environment_components.molecule_features.get_bond_pairs')
    bond_pairs.return_value = [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]

    env = GraphObsMixin()
    env.mol = mol
    env.nonring = 'nonring'
    data, nonring = env._obs()

    assert torch.sum(torch.abs(data.x - torch.tensor([[1.732, 0, 0], [0, 0, 0], [-1.732, 0, 0]]))) < 1e-2
    assert torch.all(torch.eq(data.edge_index, torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])))
    assert torch.all(torch.eq(data.edge_attr, torch.tensor([[1., 0, 1, 1, 0, 1], [1., 0, 1, 1, 0, 1], [0., 1, 1, 0, 1, 1], [0., 1, 1, 0, 1, 1]])))
    assert nonring == 'nonring'

def test_AtomTypeGraphObsMixin(mocker):
    mol = mocker.Mock()
    mol.GetConformer.return_value = mol.conf
    mol.conf.return_value = 'conf'
    mol.conf.GetPositions.return_value = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    mol.GetAtoms.return_value = ['a1', 'a2', 'a3']
    mol.GetBonds.return_value = ['b1', 'b2']

    removeHs = mocker.patch('conformer_rl.environments.environment_components.obs_mixins.Chem.rdmolops.RemoveHs')
    removeHs.return_value = mol
    atom_type = mocker.patch('conformer_rl.environments.environment_components.obs_mixins.molecule_features.atom_type_CO')
    atom_type.side_effect = atype
    bond_type = mocker.patch('conformer_rl.environments.environment_components.obs_mixins.molecule_features.bond_type')
    bond_type.side_effect = btype
    bond_pairs = mocker.patch('conformer_rl.environments.environment_components.molecule_features.get_bond_pairs')
    bond_pairs.return_value = [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]

    env = AtomTypeGraphObsMixin()
    env.mol = mol
    env.nonring = 'nonring'
    data, nonring = env._obs()

    assert torch.sum(torch.abs(data.x - torch.tensor([[1, 0], [0, 1], [1, 0]]))) < 1e-2
    assert torch.all(torch.eq(data.edge_index, torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])))
    assert torch.all(torch.eq(data.edge_attr, torch.tensor([[1., 0, 1, 1, 0, 1], [1., 0, 1, 1, 0, 1], [0., 1, 1, 0, 1, 1], [0., 1, 1, 0, 1, 1]])))
    assert nonring == 'nonring'


def test_AtomCoordsTypeGraphObsMixin(mocker):
    mol = mocker.Mock()
    mol.GetConformer.return_value = mol.conf
    mol.conf.return_value = 'conf'
    mol.conf.GetPositions.return_value = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    mol.GetAtoms.return_value = ['a1', 'a2', 'a3']
    mol.GetBonds.return_value = ['b1', 'b2']

    removeHs = mocker.patch('conformer_rl.environments.environment_components.obs_mixins.Chem.rdmolops.RemoveHs')
    removeHs.return_value = mol
    atom_type = mocker.patch('conformer_rl.environments.environment_components.obs_mixins.molecule_features.atom_type_CO')
    atom_type.side_effect = atype
    bond_type = mocker.patch('conformer_rl.environments.environment_components.obs_mixins.molecule_features.bond_type')
    bond_type.side_effect = btype
    atom_coords = mocker.patch('conformer_rl.environments.environment_components.obs_mixins.molecule_features.atom_coords')
    atom_coords.side_effect = acoords
    bond_pairs = mocker.patch('conformer_rl.environments.environment_components.molecule_features.get_bond_pairs')
    bond_pairs.return_value = [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]

    env = AtomCoordsTypeGraphObsMixin()
    env.mol = mol
    env.nonring = 'nonring'
    data, nonring = env._obs()

    assert torch.sum(torch.abs(data.x - torch.tensor([[1, 0, 1.732, 0, 0], [0, 1, 0, 0, 0], [1, 0, -1.732, 0, 0]]))) < 1e-2
    assert torch.all(torch.eq(data.edge_index, torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])))
    assert torch.all(torch.eq(data.edge_attr, torch.tensor([[1., 0, 1, 1, 0, 1], [1., 0, 1, 1, 0, 1], [0., 1, 1, 0, 1, 1], [0., 1, 1, 0, 1, 1]])))
    assert nonring == 'nonring'
    