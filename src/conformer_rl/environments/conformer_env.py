"""
Conformer_env
=============
"""
import numpy as np
import gym
import copy

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import TorsionFingerprints
from conformer_rl.utils import get_conformer_energy
from conformer_rl.config import MolConfig

import logging
from typing import List, Tuple, Mapping, Any

class ConformerEnv(gym.Env):
    """Base interface for building conformer generation environments.

    Parameters
    ----------
    mol_config : :class:`~conformer_rl.config.mol_config.MolConfig`
        Configuration object specifying molecule and parameters to be used in the environment.
    max_steps : int
        The number of steps before the end of an episode.

    Attributes
    ----------
    config : :class:`~conformer_rl.config.mol_config.MolConfig`
        Configuration object specifying molecule and parameters to be used in the environment.
    total_reward : float
        Keeps track of the total reward for the current episode.
    current_step : int
        Keeps track of the number of elapsed steps in the current episode.
    step_info : dict from str to list
        Used for keeping track of data obtained at each step of an episode for logging.
    episode_info : dict from str to Any
        Used for keeping track of data useful at the end of an episode, such as total_reward, for logging.
    

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, mol_config: MolConfig, max_steps = 200):
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
            logging.debug("Input molecule to environment should have exactly one conformer, none or more than one detected.")
            self.mol.RemoveAllConformers()
            if Chem.EmbedMolecule(self.mol, randomSeed=self.config.seed, useRandomCoords=True) == -1:
                raise Exception('Unable to embed molecule with conformer using rdkit')
        self.conf = self.mol.GetConformer()
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]

        self.reset()

    def step(self, action: Any) -> Tuple[object, float, bool, dict]:
        """Simulates one iteration of the environment.

        Updates the environment with the input action, and calculates the current observation,
        reward, done, and info.

        Parameters
        ----------
        action : Any, depending on implementation of :meth:`~ConformerEnv._step()`
            The action to be taken by the environment.

        Returns
        -------
        obs : Any, depending on implementation of :meth:`~ConformerEnv._obs()`
            An object reflecting the current configuration of the environment/molecule.
        reward : float
            The reward calculated given the current configuration of the environment.
        done : bool
            Whether or not the current episode has finished.
        info : dict
            Information about the current step and episode of the environment, to be used for logging.

        Notes
        -----
        Logged parameters:

        * reward (float): the reward for the current step
        """
        self.action = action

        self._step(action)
        self.current_step += 1

        assert(self.mol.GetNumConformers() == 1)

        obs = self._obs()
        reward = self._reward()
        self.step_info['reward'] = reward
        self.total_reward += reward
        done = self._done()
        info = copy.deepcopy(self._info())

        logging.info(f"step {self.current_step} reward {reward}")


        return obs, reward, done, info
        
    def reset(self) -> object:
        """Resets the environment and returns the observation of the environment.
        """
        logging.info("reset called")
        # reset environment state variables
        self.total_reward = 0
        self.current_step = 0

        self.step_info = {}
        self.episode_info['mol'] = Chem.Mol(self.mol)
        self.episode_info['mol'].RemoveAllConformers()

        obs = self._obs()

        return obs

    def _step(self, action: Any) -> None:
        """Does not modify molecule.

        Notes
        -----
        Logged parameters:

        * conf: the current generated conformer is saved to the episodic mol object.
        """
        self.episode_info['mol'].AddConformer(self.conf, assignId=True)

    def _obs(self) -> Chem.rdchem.Mol:
        """Returns the current molecule.
        """
        return self.mol

    def _reward(self) -> float:
        """Returns :math:`e^{-1 * energy}` where :math:`energy` is the
        energy of the current conformer of the molecule.

        Notes
        -----
        Logged parameters:

        * energy (float): the energy of the current conformer
        """
        energy = get_conformer_energy(self.mol)
        reward =  np.exp(-1. * energy)

        self.step_info['energy'] = energy
        return reward

    def _done(self) -> bool:
        """Returns true if the current number of elapsed steps has exceeded
        the max number of steps per episode.
        """
        return self.current_step >= self.max_steps

    def _info(self) -> Mapping[str, Mapping[str, Any]]:
        """Returns a dict wrapping `episode_info` and `step_info`.

        Notes
        -----
        Logged parameters:
        
        * total_reward (float): total reward of the episode is updated
        """
        info = {}

        if self._done():
            self.episode_info["total_rewards"] = self.total_reward
            info['episode_info'] = self.episode_info

        info['step_info'] = self.step_info

        return info

        