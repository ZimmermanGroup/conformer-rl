from conformer_rl.environments.environment_components.reward_mixins import GibbsRewardMixin, GibbsPruningRewardMixin, GibbsEndPruningRewardMixin, GibbsLogPruningRewardMixin
from rdkit import Chem
import numpy as np

def test_GibbsRewardMixin(mocker):
    super = mocker.patch('conformer_rl.environments.environment_components.reward_mixins.super')
    get_energy = mocker.patch('conformer_rl.environments.environment_components.reward_mixins.get_conformer_energy')
    get_energy.return_value = 11
    env = GibbsRewardMixin()
    env.episode_info = {}
    env.step_info = {}
    env.reset()
    assert env.seen == set()
    assert env.repeats == 0
    assert env.episode_info['repeats'] == 0
    super.assert_called_once()

    env.action = 'action'
    env.config = mocker.Mock()
    env.config.E0 = 12.5
    env.config.tau = 500
    env.config.Z0 = 5
    env.mol = 'mol'

    reward = env._reward()
    assert abs(reward - 0.90595) < 1e-4
    assert env.repeats == 0
    assert env.step_info['energy'] == 11

    env.seen.add('action')
    reward = env._reward()
    assert env.repeats == 1
    assert reward == 0

def test_GibbsEndPruningRewardMixin(mocker):
    super = mocker.patch('conformer_rl.environments.environment_components.reward_mixins.super')
    get_energy = mocker.patch('conformer_rl.environments.environment_components.reward_mixins.get_conformer_energy')
    get_energy.return_value = 11
    env = GibbsEndPruningRewardMixin()
    env.episode_info = {}
    env.step_info = {}
    mol = Chem.MolFromSmiles('CCCCCCCC')
    Chem.AllChem.EmbedMolecule(mol)
    env.mol = mol
    env.conf = mol.GetConformer()
    env.reset()

    assert env.backup_mol.GetNumConformers() == 0
    assert env.backup_mol.GetNumAtoms() == 8
    super.assert_called_once()

    env.config = mocker.Mock()
    env.config.E0 = 12.5
    env.config.tau = 500
    env.config.Z0 = 5
    env._done = mocker.Mock()
    env._done.return_value = True
    penalty = mocker.patch('conformer_rl.environments.environment_components.reward_mixins.GibbsEndPruningRewardMixin._pruning_penalty')
    penalty.return_value = 1.


    reward = env._reward()
    assert abs(reward - (-0.09405)) < 1e-4
    assert env.step_info['energy'] == 11
    assert env.backup_mol.GetNumConformers() == 1

def test_GibbsEndPruning_penalty(mocker):
    mocker.patch('conformer_rl.environments.environment_components.reward_mixins.prune_conformers')
    conf_energies = mocker.patch('conformer_rl.environments.environment_components.reward_mixins.get_conformer_energies')
    conf_energies.side_effect = [
        np.array([1., 2, 3, 5, 7, 11, 13]),
        np.array([1., 3, 7, 11])
    ]

    env = GibbsEndPruningRewardMixin()
    env.backup_mol = None
    env.config = mocker.Mock()
    env.config.E0 = 12.5
    env.config.tau = 500
    env.config.Z0 = 5

    penalty = env._pruning_penalty()
    assert abs(penalty - 8207.8485) < 1e-2

def test_GibbsPruningRewardMixin(mocker):
    super = mocker.patch('conformer_rl.environments.environment_components.reward_mixins.super')
    get_energy = mocker.patch('conformer_rl.environments.environment_components.reward_mixins.get_conformer_energy')
    get_energy.return_value = 11
    env = GibbsEndPruningRewardMixin()
    env.episode_info = {}
    env.step_info = {}
    mol = Chem.MolFromSmiles('CCCCCCCC')
    Chem.AllChem.EmbedMolecule(mol)
    env.mol = mol
    env.conf = mol.GetConformer()
    env.reset()

    assert env.backup_mol.GetNumConformers() == 0
    assert env.backup_mol.GetNumAtoms() == 8
    super.assert_called_once()

    env.config = mocker.Mock()
    env.config.E0 = 12.5
    env.config.tau = 500
    env.config.Z0 = 5
    env._done = mocker.Mock()
    env._done.return_value = True
    penalty = mocker.patch('conformer_rl.environments.environment_components.reward_mixins.GibbsEndPruningRewardMixin._pruning_penalty')
    penalty.return_value = 1.


    reward = env._reward()
    assert abs(reward - (-0.09405)) < 1e-4
    assert env.step_info['energy'] == 11
    assert env.backup_mol.GetNumConformers() == 1