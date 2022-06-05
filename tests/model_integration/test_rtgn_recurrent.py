import numpy as np
import torch

from conformer_rl import utils
from conformer_rl.agents import PPORecurrentAgent
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGNRecurrent

from conformer_rl.molecule_generation.generate_lignin import generate_lignin
from conformer_rl.molecule_generation.generate_molecule_config import config_from_rdkit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.INFO)

def test_rtgn_recurrent(mocker):
    utils.set_one_thread()

    mol_config = config_from_rdkit(generate_lignin(2), num_conformers=8, calc_normalizers=True)

    config = Config()
    config.tag = 'example1'
    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)
    # Batch Hyperparameters
    config.num_workers = 2
    config.rollout_length = 2
    config.recurrence = 1
    config.optimization_epochs = 1
    config.max_steps = 24
    config.save_interval = 8
    config.eval_interval = 8
    config.eval_episodes = 1
    config.mini_batch_size = 4

    # Coefficient Hyperparameters
    lr = 5e-6 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)

    # Task Settings
    config.train_env = Task('GibbsScorePruningEnv-v0', concurrency=False, num_envs=config.num_workers, seed=np.random.randint(0,1e5), mol_config=mol_config)
    config.eval_env = Task('GibbsScorePruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=mol_config)
    config.curriculum = None

    agent = PPORecurrentAgent(config)
    agent.run_steps()