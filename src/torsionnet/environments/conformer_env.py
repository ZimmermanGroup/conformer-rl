import numpy as np
import gym
import copy

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import TorsionFingerprints
from torsionnet.utils import get_conformer_energy

import logging
from typing import List, Tuple, Mapping, Any

class ConformerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mol_config: object, max_steps = 200):
        super(ConformerEnv, self).__init__()
        logging.info('initializing conformer environment')
        self.config = copy.deepcopy(mol_config)
        self.max_steps = max_steps
        self.total_reward = 0
        self.current_step = 0

        self.step_info = {}
        self.episode_info = {}

        self.mol = self.config.mol

        # set mol to have exactly one conformer
        if self.mol.GetNumConformers() != 1:
            logging.warn("Input molecule to environment should have exactly one conformer, none or more than one detected.")
            self.mol.RemoveAllConformers()
            if Chem.EmbedMolecule(self.mol, randomSeed=self.config.seed, useRandomCoords=True) == -1:
                raise Exception('Unable to embed molecule with conformer using rdkit')
        self.conf = self.mol.GetConformer()

        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]

        self.reset()

    def step(self, action: Any) -> Tuple[object, float, bool, dict]:
        self.action = action
        logging.debug(str(action))

        self._step(action)
        self.current_step += 1

        assert(self.mol.GetNumConformers() == 1)

        obs = self._obs()
        reward = self._reward()
        self.total_reward += reward
        done = self._done()
        info = copy.deepcopy(self._info())

        logging.info(f"step {self.current_step} reward {reward}")
        self.step_info['reward'] = reward

        return obs, reward, done, info
        
    def reset(self) -> object:
        logging.info("reset called")
        # reset environment state variables
        self.total_reward = 0
        self.current_step = 0

        self.step_info = {}
        self.episode_info['mol'] = Chem.Mol(self.mol)
        self.episode_info['mol'].RemoveAllConformers()

        obs = self._obs()

        return obs

    def render(self, mode='human') -> None:
        pass

    def _step(self, action: Any) -> None:
        self.episode_info['mol'].AddConformer(self.conf, assignId=True)

    def _obs(self) -> Chem.rdchem.Mol:
        return self.mol

    def _reward(self) -> float:
        energy = get_conformer_energy(self.mol)
        reward =  np.exp(-1. * energy)

        self.step_info['energy'] = energy
        return reward

    def _done(self) -> bool:
        return self.current_step >= self.max_steps

    def _info(self) -> Mapping[str, Mapping[str, Any]]:
        if self._done():
            self.episode_info["total_rewards"] = self.total_reward

        return {'episode_info': self.episode_info, 'step_info': self.step_info}

        