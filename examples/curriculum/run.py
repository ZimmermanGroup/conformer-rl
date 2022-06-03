import numpy as np
import torch
import pickle

from conformer_rl import utils
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGNRecurrent

from conformer_rl.molecule_generation.generate_alkanes import generate_branched_alkane
from conformer_rl.molecule_generation.generate_molecule_config import config_from_rdkit
from conformer_rl.agents import PPORecurrentAgent

from conformer_rl.agents.curriculum_agent_mixin import ExternalCurriculumAgentMixin
import curriculum_env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                                                                                                                                                                                     
import logging
logging.basicConfig(level=logging.DEBUG)


class CurriculumAgent(ExternalCurriculumAgentMixin, PPORecurrentAgent):
    pass

if __name__ == '__main__':
    utils.set_one_thread()

    mol_configs = [config_from_rdkit(generate_branched_alkane(i), num_conformers=6, calc_normalizers=True) for i in range(8, 16)]

    config = Config()
    config.tag = 'curriculum_test'
    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)

    # Batch Hyperparameters
    config.rollout_length = 2
    config.recurrence = 1
    config.optimization_epochs = 1
    config.max_steps = 96
    config.save_interval = 8
    config.eval_interval = 8
    config.eval_episodes = 1
    config.mini_batch_size = 4

    # curriculum Hyperparameters
    config.curriculum_agent_buffer_len = 4
    config.curriculum_agent_reward_thresh = 0.
    config.curriculum_agent_success_rate = 0.
    config.curriculum_agent_fail_rate = -1.


    # Coefficient Hyperparameters
    lr = 5e-6 * np.sqrt(2)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)

    # Task Settings
    config.train_env = Task('CurriculumEnv-v0', concurrency=False, num_envs=2, seed=np.random.randint(0,1e5), mol_configs=mol_configs)
    config.eval_env = Task('GibbsScorePruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=mol_configs[-1])
    config.curriculum = None

    agent = CurriculumAgent(config)
    agent.run_steps()