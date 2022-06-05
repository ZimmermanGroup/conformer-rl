import numpy as np
import torch

from conformer_rl import utils
from conformer_rl.agents import PPORecurrentAgent
from conformer_rl.config import Config
from conformer_rl.environments import Task

from conformer_rl.molecule_generation.generate_alkanes import generate_branched_alkane
from conformer_rl.molecule_generation.generate_molecule_config import config_from_rdkit

import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    utils.set_one_thread()

    # Create config object
    mol = generate_branched_alkane(14)
    mol_config = config_from_rdkit(mol, num_conformers=200, calc_normalizers=True, save_file='alkane')

    # Create agent training config object
    config = Config()
    config.tag = 'example1'

    # Configure Environment
    config.train_env = Task('GibbsScorePruningEnv-v0', concurrency=True, num_envs=5, seed=np.random.randint(0,1e5), mol_config=mol_config)
    config.eval_env = Task('GibbsScorePruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=mol_config)
    config.eval_episodes=10000

    agent = PPORecurrentAgent(config)
    agent.run_steps()