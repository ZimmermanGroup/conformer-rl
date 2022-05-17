import numpy as np
import torch

from conformer_rl import utils
from conformer_rl.agents import PPOAgent
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGN

from conformer_rl.molecule_generation.generate_alkanes import branched_alkane_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    utils.set_one_thread()

    mol_config = branched_alkane_config(15)

    config = Config()
    config.tag = 'example2'
    config.network = RTGN(6, 128, edge_dim=6, node_dim=5).to(device)

    # Training Hyperparameters
    lr = 5e-6 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)

    # Task Settings
    config.train_env = Task('GibbsScorePruningEnv-v0', concurrency=True, num_envs=config.num_workers, seed=np.random.randint(0,1e5), mol_config=mol_config, max_steps=200)
    config.eval_env = Task('GibbsScorePruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=mol_config, max_steps=200)

    agent = PPOAgent(config)
    agent.run_steps()