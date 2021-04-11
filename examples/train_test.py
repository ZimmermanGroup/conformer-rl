import numpy as np
import random
import torch

from torsionnet.utils import *
from torsionnet.agents import PPORecurrentAgent
from torsionnet.config import Config
from torsionnet.environments import Task
from torsionnet.models import RTGNBatch
from torsionnet.generate_molecule import DIFF, XORGATE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ppo_feature(tag, model):
    config = Config()
    config.tag = tag

    # Global Settings
    config.network = model
    config.hidden_size = model.dim

    # Task Settings
    config.train_env = Task('ConfEnv-v1', concurrency=True, num_envs=config.num_workers, seed=random.randint(0,1e5), mol_config=DIFF, max_steps=4)
    config.eval_env = Task('ConfEnv-v1', seed=random.randint(0,7e4), mol_config=DIFF, max_steps=4)

    return PPORecurrentAgent(config)


if __name__ == '__main__':
    nnet = RTGNBatch(6, 128, edge_dim=6, point_dim=5)
    nnet.to(device)
    set_one_thread()
    tag = 'Diff-v2'
    agent = ppo_feature(tag=tag, model=nnet)
    agent.run_steps()
