import conformer_rl
from conformer_rl.environments.environment_components.action_mixins import ContinuousActionMixin, DiscreteActionMixin
from rdkit import Chem

def test_continuous(mocker):
    setDihedrals = mocker.patch('conformer_rl.environments.environment_components.action_mixins.Chem.rdMolTransforms.SetDihedralDeg')
    MMFFOptimize = mocker.patch('conformer_rl.environments.environment_components.action_mixins.Chem.AllChem.MMFFOptimizeMolecule')

    env = ContinuousActionMixin()
    mol = Chem.MolFromSmiles('CCCCCCCC')
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol)
    env.mol = mol
    env.conf = mol.GetConformer()
    env.episode_info = {}
    env.episode_info['mol'] = mol
    env.nonring = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    env._step([123, 456, 789])
    assert setDihedrals.call_count == 3
    args_list = setDihedrals.call_args_list
    assert args_list == [
        mocker.call(env.conf, 1, 2, 3, 4, 123.),
        mocker.call(env.conf, 5, 6, 7, 8, 456.),
        mocker.call(env.conf, 9, 10, 11, 12, 789.)
    ]

    MMFFOptimize.assert_called_once()

    assert env.episode_info['mol'].GetNumConformers() == 2

def test_discrete(mocker):
    setDihedrals = mocker.patch('conformer_rl.environments.environment_components.action_mixins.Chem.rdMolTransforms.SetDihedralDeg')
    MMFFOptimize = mocker.patch('conformer_rl.environments.environment_components.action_mixins.Chem.AllChem.MMFFOptimizeMolecule')

    env = DiscreteActionMixin()
    mol = Chem.MolFromSmiles('CCCCCCCC')
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol)
    env.mol = mol
    env.conf = mol.GetConformer()
    env.episode_info = {}
    env.episode_info['mol'] = mol
    env.nonring = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    env._step([0, 2, 3])
    assert setDihedrals.call_count == 3
    args_list = setDihedrals.call_args_list
    assert args_list == [
        mocker.call(env.conf, 1, 2, 3, 4, -180.),
        mocker.call(env.conf, 5, 6, 7, 8, -60.),
        mocker.call(env.conf, 9, 10, 11, 12, 0.)
    ]

    MMFFOptimize.assert_called_once()

    assert env.episode_info['mol'].GetNumConformers() == 2

