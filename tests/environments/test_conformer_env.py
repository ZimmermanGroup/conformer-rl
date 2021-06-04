from conformer_rl.environments.conformer_env import ConformerEnv
from conformer_rl.molecule_generation.molecules import test_alkane
from rdkit import Chem
import pytest

def test_conformer_env(mocker):
    step = mocker.patch('conformer_rl.environments.conformer_env.ConformerEnv._step')
    config = test_alkane()
    env = ConformerEnv(config)

    assert env.mol.GetNumAtoms() == 47
    mol = Chem.RemoveHs(env.mol)
    assert mol.GetNumAtoms() == 15
    assert env.mol.GetNumConformers() == 1

    obs, reward, done, info = env.step(180)
    assert obs == env._obs()
    assert reward == env._reward()
    assert done == env._done()
    assert 'episode_info' in info
    assert 'step_info' in info
    step.assert_called_with(180)

def test_conformer_env1(mocker):
    config = test_alkane()
    env = ConformerEnv(config)

    assert env.mol.GetNumAtoms() == 47
    mol = Chem.RemoveHs(env.mol)
    assert mol.GetNumAtoms() == 15
    assert env.mol.GetNumConformers() == 1

    obs, reward, done, info = env.step(180)
    for i in range(201):
        obs, reward, done, info = env.step(180)

    assert 'total_rewards' in info['episode_info']
    assert env.episode_info['mol'].GetNumConformers() == 202

    env.reset()
    assert env.total_reward == 0
    assert env.mol.GetNumConformers() == 1
    assert env.episode_info['mol'].GetNumConformers() == 0

def test_step(mocker):
    energy = mocker.patch('conformer_rl.environments.conformer_env.get_conformer_energy')
    energy.return_value = 5

    config = test_alkane()
    env = ConformerEnv(config)

    obs, reward, done, info = env.step(180)
    assert abs(reward - 0.006738) < 1e-4


def test_exception(mocker):
    embed = mocker.patch('conformer_rl.environments.conformer_env.Chem.EmbedMolecule')
    embed.return_value = -1
    config = test_alkane()
    with pytest.raises(Exception):
        env = ConformerEnv(config)



    




    
