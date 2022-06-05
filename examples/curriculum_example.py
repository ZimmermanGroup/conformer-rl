import numpy as np
import torch
import pickle

from conformer_rl import utils
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGNRecurrent

from conformer_rl.molecule_generation.generate_alkanes import generate_branched_alkane
from conformer_rl.molecule_generation.generate_molecule_config import config_from_rdkit
from conformer_rl.agents import PPORecurrentExternalCurriculumAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                                                                                                                                                                                     
import logging
logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':
    utils.set_one_thread()

    # Create mol_configs for the curriculum
    mol_configs = [config_from_rdkit(generate_branched_alkane(i), num_conformers=200, calc_normalizers=True) for i in range(8, 16)]
    eval_mol_config = config_from_rdkit(generate_branched_alkane(16), num_conformers=200, calc_normalizers=True)

    config = Config()
    config.tag = 'curriculum_test'
    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)

    # Batch Hyperparameters
    config.max_steps = 100000

    # training Hyperparameters
    lr = 5e-6 * np.sqrt(10)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)

    # Task Settings
    config.train_env = Task('GibbsScorePruningCurriculumEnv-v0', concurrency=True, num_envs=10, seed=np.random.randint(0,1e5), mol_configs=mol_configs)
    config.eval_env = Task('GibbsScorePruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=eval_mol_config)
    config.eval_interval = 20000

    # curriculum Hyperparameters
    config.curriculum_agent_buffer_len = 20
    config.curriculum_agent_reward_thresh = 0.4
    config.curriculum_agent_success_rate = 0.7
    config.curriculum_agent_fail_rate = 0.2

    agent = PPORecurrentExternalCurriculumAgent(config)
    agent.run_steps()