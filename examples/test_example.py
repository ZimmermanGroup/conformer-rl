import numpy as np
import torch
import pickle

from conformer_rl import utils
from conformer_rl.agents import PPORecurrentAgent
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGNRecurrent


from conformer_rl.molecule_generation.generate_alkanes import generate_branched_alkane
from conformer_rl.molecule_generation.generate_molecule_config import config_from_rdkit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                                                                                                                                                                                 
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    utils.set_one_thread()

    mol_config = config_from_rdkit(generate_branched_alkane(10), num_conformers=4, calc_normalizers=True, save_file='10_alkane')
    with open('10_alkane.pkl', 'rb') as file:
        mol_config = pickle.load(file)

    config = Config()
    config.tag = 'test_example'
    # config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)
    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)
    # Batch Hyperparameters
    config.rollout_length = 2
    config.recurrence = 1
    config.optimization_epochs = 1
    config.max_steps = 24
    config.save_interval = 8
    config.eval_interval = 8
    config.eval_episodes = 1
    config.mini_batch_size = 4

    # Coefficient Hyperparameters
    lr = 5e-6 * np.sqrt(2)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)

    # Task Settings
    config.train_env = Task('GibbsScoreEnv-v0', concurrency=True, num_envs=2, seed=np.random.randint(0,1e5), mol_config=mol_config)
    config.eval_env = Task('GibbsScorePruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=mol_config)
    config.curriculum = None

    agent = PPORecurrentAgent(config)
    agent.run_steps()