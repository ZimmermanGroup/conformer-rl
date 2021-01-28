import numpy as np
import random
import argparse
import torch

from main import mkdir
from main import PPORecurrentAgent
from main import Config
from main import Task, Curriculum
from main import RTGNBatchXorgate

from generate_molecule import DIFF, XORGATE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ppo_feature(tag, model):
    config = Config()
    config.tag = tag

    # Global Settings
    config.network = model
    config.hidden_size = model.dim

    # Batch Hyperparameters
    config.num_workers = 1 #20
    config.rollout_length = 4 #20
    config.recurrence = 1 #2
    config.optimization_epochs = 1 #4
    config.max_steps = 10000000
    config.save_interval = 0 #config.num_workers*200*5
    config.eval_interval = 0 #config.num_workers*200*5
    config.eval_episodes = 2
    config.mini_batch_size = 2 #40

    # Coefficient Hyperparameters
    lr = 5e-6 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.discount = 0.9999
    config.use_gae = True
    config.gae_tau = 0.95
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.001
    config.gradient_clip = 0.5
    config.ppo_ratio_clip = 0.2

    # Task Settings
    config.train_env = Task('XorgateEnv-v0', concurrency=False, num_envs=config.num_workers, seed=random.randint(0,1e5), mol_config=XORGATE, max_steps=8)
    config.eval_env = Task('XorgateEnv-v0', seed=random.randint(0,7e4), mol_config=XORGATE, max_steps=8)
    config.curriculum = None
    # config.curriculum = Curriculum(min_length=config.num_workers)

    return PPORecurrentAgent(config)


if __name__ == '__main__':
    nnet = RTGNBatchXorgate(6, 128, edge_dim=6, point_dim=5)
    nnet.to(device)
    mkdir('train_data')
    mkdir('model_data')
    mkdir('molecule_data')
    # set_one_thread()
    # select_device(0)
    tag = 'XorGate-v0'
    agent = ppo_feature(tag=tag, model=nnet)
    agent.run_steps()
