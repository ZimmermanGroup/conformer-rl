import numpy as np
import torch

from conformer_rl import utils
from conformer_rl.agents import PPORecurrentAgent
from conformer_rl.models import RTGNRecurrent
from conformer_rl.config import Config
from conformer_rl.environments import Task

from conformer_rl.molecule_generation.generate_alkanes import generate_branched_alkane
from conformer_rl.molecule_generation.generate_molecule_config import config_from_rdkit

import logging
logging.basicConfig(level=logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# import the custom created environment to run the gym register script
import custom_env

if __name__ == '__main__':
    utils.set_one_thread()

    # Create config object
    mol = generate_branched_alkane(14)
    mol_config = config_from_rdkit(mol, calc_normalizers=True, ep_steps=200, save_file='alkane')

    # Create agent training config object
    config = Config()

    # set the tag to reflect the custom environment
    config.tag = 'atom_type_env'

    # Update the neural network node_dim to equal 2
    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=2).to(device)

    # Set the environment to the test env
    config.train_env = Task('TestEnv-v0', concurrency=True, num_envs=5, seed=np.random.randint(0,1e5), mol_config=mol_config, max_steps=200)
    config.eval_env = Task('TestEnv-v0', seed=np.random.randint(0,7e4), mol_config=mol_config, max_steps=200)
    config.eval_episodes=10000

    agent = PPORecurrentAgent(config)
    agent.run_steps()