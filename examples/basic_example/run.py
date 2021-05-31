import numpy as np
import torch

from conformer_rl.agents import PPORecurrentAgent
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGNRecurrent

from conformer_rl.molecule_generation import branched_alkane

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # Set up the environment.
    alkane_env_config = branched_alkane(num_atoms=18)
    training_env = Task('GibbsScorePruningEnv-v0', concurrency=True, num_envs=20, mol_config=alkane_env_config, max_steps=200)


    # initialize Config
    config = Config()
    config.tag = 'tutorial'
    config.train_env = training_env

    # Set up neural network
    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)
    config.num_workers = 20

    # Logging Parameters
    config.save_interval = 20000
    config.data_dir = 'data'
    config.use_tensorboard = True

    # Set up evaluation
    config.eval_env = Task('GibbsScorePruningEnv-v0', num_envs=1, mol_config=alkane_env_config, max_steps=200)
    config.eval_interval = 20000
    config.eval_episodes = 2

    # Batch Hyperparameters
    config.rollout_length = 20
    config.recurrence = 5
    config.optimization_epochs = 4
    config.max_steps = 10000000
    config.mini_batch_size = 50

    # Coefficient Hyperparameters
    lr = 5e-6 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.discount = 0.9999
    config.use_gae = True
    config.gae_lambda = 0.95
    config.entropy_weight = 0.001
    config.value_loss_weight = 0.25
    config.gradient_clip = 0.5
    config.ppo_ratio_clip = 0.2

    # run the agent
    agent = PPORecurrentAgent(config)
    agent.run_steps()