"""
Reward_mixins
=============

Pre-built reward handlers.
"""
import numpy as np
from rdkit import Chem

from conformer_rl.utils import get_conformer_energies, get_conformer_energy, prune_conformers, prune_last_conformer

KB = 0.001985875 # Boltzmann constant in kcal/(mol * K)

# Gibbs Score
class GibbsRewardMixin:
    """Implements the Gibbs Score reward [1]_, but distance metric between conformers are judged by
    whether the conformers were produced by the same action input, instead of with TFD (Torsional Fingerprint Deviation).

    References
    ----------
    .. [1] `TorsionNet paper <https://arxiv.org/abs/2006.07078>`_
    """
    def reset(self):
        self.seen = set()
        self.repeats = 0
        self.episode_info['repeats'] = 0
        return super().reset()

    def _reward(self) -> float:
        """
        Notes
        -----
        Logged parameters:

        * energy (float): the energy of the current conformer
        * repeat (int): total number of repeated actions so far in the episode
        """
        config = self.config

        energy = get_conformer_energy(self.mol)
        self.step_info['energy'] = energy

        if tuple(self.action) in self.seen:
            self.repeats += 1
            self.episode_info['repeat'] = self.repeats
            return 0.
        else:
            self.seen.add(tuple(self.action))
            reward = np.exp(-1. * (energy - config.E0) / (KB * config.tau)) / config.Z0
            return reward

class GibbsEndPruningRewardMixin:
    """Implements the Gibbs Score reward [1]_, except overly similar conformers are only pruned at the end of an episode
    and therefore is only reflected in the final reward in each episode.

    """
    def reset(self):
        self.backup_mol = Chem.Mol(self.mol)
        self.backup_mol.RemoveAllConformers()
        return super().reset()

    def _reward(self) -> float:
        """
        Notes
        -----
        Logged parameters:

        * energy (float): the energy of the current conformer
        """
        config = self.config

        self.backup_mol.AddConformer(self.conf, assignId=True)

        energy = get_conformer_energy(self.mol)
        self.step_info['energy'] = energy

        reward = np.exp(-1. * (energy - config.E0) / (KB * config.tau)) / config.Z0

        if self._done():
            reward -= self._pruning_penalty()
        return reward

    def _pruning_penalty(self):
        config = self.config

        before_total = np.exp(-1.0 * (get_conformer_energies(self.backup_mol) - config.E0) / (KB * config.tau)).sum() / config.Z0
        self.backup_mol = prune_conformers(self.backup_mol, config.pruning_thresh)
        after_total = np.exp(-1.0 * (get_conformer_energies(self.backup_mol) - config.E0) / (KB * config.tau)).sum() / config.Z0
        return before_total - after_total

class GibbsPruningRewardMixin:
    """Implements the Gibbs Score reward [1]_.


    """
    def reset(self):
        self.backup_mol = Chem.Mol(self.mol)
        self.backup_mol.RemoveAllConformers()
        self.backup_energys = []
        return super().reset()

    def _reward(self) -> float:
        """
        Notes
        -----
        Logged parameters:

        * energy (float): the energy of the current conformer
        """
        config = self.config
        
        self.backup_mol.AddConformer(self.conf, assignId=True)
        energy = get_conformer_energy(self.mol)
        self.step_info['energy'] = energy
        self.backup_energys.append(energy)

        self._prune_conformers()
        total_reward = np.exp(-1.0 * (np.array(self.backup_energys) - config.E0) / (KB * config.tau)).sum() / config.Z0
        rew = total_reward - self.total_reward
        return rew

    def _prune_conformers(self):
        config = self.config
        self.backup_mol, self.backup_energys = prune_last_conformer(self.backup_mol, config.pruning_thresh, self.backup_energys)
        assert self.backup_mol.GetNumConformers() == len(self.backup_energys)

class GibbsLogPruningRewardMixin(GibbsPruningRewardMixin):
    """Implements the log of the Gibbs Score reward [1]_.

    """
    def _reward(self) -> float:
        """
        Notes
        -----
        Logged parameters:
        
        * energy (float): the energy of the current conformer
        """
        config = self.config

        self.backup_mol.AddConformer(self.conf, assignId=True)
        energy = get_conformer_energy(self.mol)
        self.step_info['energy'] = energy
        self.backup_energys.append(energy)

        self._prune_conformers()

        total_reward = np.log(np.exp(-1.0 * (np.array(self.backup_energys) - config.E0) / (KB * config.tau)).sum() / config.Z0)
        if not np.isfinite(total_reward):
            total_reward = np.finfo(np.float64).eps

        rew = total_reward - self.total_reward
        return rew