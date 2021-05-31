import numpy as np
import torch

from conformer_rl import utils
from conformer_rl.agents import A2CRecurrentAgent, A2CAgent
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGNRecurrent, RTGN, RTGNGat

from conformer_rl.molecule_generation import xorgate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    utils.set_one_thread()

    mol_config = xorgate(2, 3)

    config = Config()
    config.tag = 'example1'
    # config.network = RTGN(6, 128, edge_dim=6, node_dim=5).to(device)
    config.network = RTGNGat(6, 128, node_dim=5).to(device)
    # Batch Hyperparameters
    config.num_workers = 2
    config.rollout_length = 2
    config.recurrence = 1
    config.max_steps = 16
    config.save_interval = 8
    config.eval_interval = 8
    config.eval_episodes = 1

    # Coefficient Hyperparameters
    lr = 5e-6 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)

    # Task Settings
    config.train_env = Task('GibbsScorePruningEnv-v0', concurrency=False, num_envs=config.num_workers, seed=np.random.randint(0,1e5), mol_config=mol_config, max_steps=4)
    config.eval_env = Task('GibbsScorePruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=mol_config, max_steps=20)
    config.curriculum = None

    agent = A2CAgent(config)
    agent.run_steps()