import numpy as np
import gym

from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints

import os.path
import multiprocessing

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance, NormalizeScale, Center, NormalizeRotation

import logging

from ...utils import ConformerGeneratorCustom, print_torsions

confgen = ConformerGeneratorCustom(max_conformers=1,
                             rmsd_threshold=None,
                             force_field='mmff',
                             pool_multiplier=1)

class ConformerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mol_config, max_steps=200):
        super(ConformerEnv, self).__init__()
        print('initializing conformer environment')
        self.episode_count = 0
        self.mol_config = mol_config
        self.max_steps = max_steps
        self.total_reward = 0
        self.current_step = 0

        self.reset()

    def step(self, action):
        self.action = action
        self.current_step += 1

        self._handle_action()
        self._update_memory()


        obs = self._get_obs()
        reward = self._get_reward()
        self.total_reward += reward
        done = self._get_done()
        info = self._get_info()

        print("episode", self.episode_count, "step", self.current_step, "reward", reward)
        self.render()

        return obs, reward, done, info
        
    def reset(self):
        print("reset called")
        print("epsiode", self.episode_count, "total reward", self.total_reward)
        self.episode_count += 1
        self.total_reward = 0
        self.current_step = 0

        self.selected_config = self._select_molecule()
        self._parse_molecule()

        obs = self._get_obs()
        return obs

    def _get_done(self):
        return (self.current_step > self.max_steps)

    def _get_info(self):
        info = {}
        done = self._get_done()
        if (done):
            info['episodic_return'] = self.total_reward
        else:
            info['episodic_return'] = None
        
        return info

    def render(self, mode='human'):
        return self.molecule

    def _select_molecule(self):
        return self.mol_config[0]

    def _parse_molecule(self):
        self.molecule = self.selected_config.molecule
        self.conf = self.molecule.GetConformer(id=0)
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.molecule)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]

    def _get_obs(self):
        raise NotImplementedError

    def _get_reward(self):
        raise NotImplementedError

    def _handle_action(self):
        raise NotImplementedError

    def _update_memory(self):
        pass


        