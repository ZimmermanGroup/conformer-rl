import numpy as np
import torch
import gym

import conformer_rl
from conformer_rl.environments import ConformerEnv
from conformer_rl.utils import get_conformer_energy

class CustomEnv1(ConformerEnv):
    def __init__(self, mol_config: conformer_rl.config.MolConfig, max_steps: int):
        super().__init__(mol_config, max_steps)

        # ensure that mol_config has energy_thresh attribute
        if not hasattr(mol_config,'energy_thresh'):
            raise Exception('mol_config must have energy_thresh attribute to use CustomEnv1')

        # set the energy threshold
        self.energy_thresh = mol_config.energy_thresh
        self.confs_below_threshold = 0

    def reset(self):
        self.confs_below_threshold = 0
        return super().reset()

    def _reward(self):
        energy = get_conformer_energy(self.mol)
        self.step_info['energy'] = energy # log energy

        reward = 1. if energy < self.energy_thresh else 0.
        if energy < self.energy_thres:
            self.confs_below_threshold += 1
            self.episode_info['confs_below_threshold'] = self.confs_below_threshold
        return reward

gym.register(
    id='CustomEnv-v0',
    entry_point='custom_env:CustomEnv'
)



