import numpy as np
import torch

from torsionnet import utils
from torsionnet.agents import PPORecurrentAgent
from torsionnet.config import Config
from torsionnet.environments import Task
from torsionnet.models import RTGNRecurrent

from torsionnet.molecules import test_alkane

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    utils.set_one_thread()

    mol_config = test_alkane()

    config = Config()
    config.tag = 'example1'
    config.network = RTGNRecurrent(6, 128, edge_dim=6, point_dim=5).to(device)
    # Batch Hyperparameters
    config.num_workers = 20
    config.rollout_length = 20
    config.recurrence = 5
    config.optimization_epochs = 4
    config.max_steps = 10000000
    config.save_interval = config.num_workers*200*5
    config.eval_interval = config.num_workers*200*5
    config.eval_episodes = 2
    config.mini_batch_size = 50

    # Coefficient Hyperparameters
    lr = 5e-6 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)

    # Task Settings
    config.train_env = Task('GibbsScorePruningEnv-v0', concurrency=True, num_envs=config.num_workers, seed=np.random.randint(0,1e5), mol_config=mol_config, max_steps=200)
    config.eval_env = Task('GibbsScorePruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=mol_config, max_steps=200)
    config.curriculum = None

    agent = PPORecurrentAgent(config)
    agent.run_steps()