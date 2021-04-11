from main.utils.chem_utils import get_conformer_energies
import numpy as np
import logging

from rdkit import Chem

from .conformer_env import ConformerEnv
from ...utils import prune_conformers, prune_last_conformer, get_conformer_energies

# Gibbs Score
class GibbsRewardMixin(ConformerEnv):
    def _parse_molecule(self):
        super()._parse_molecule()
        
        self.seen = set()
        self.repeats = 0

        mol = self.selected_config
        if mol.inv_temp is not None:
            self.temp_0 = self.mol.inv_temp
        else:
            self.temp_0 = 1
        self.standard_energy = mol.standard
        if mol.total is not None:
            self.total = mol.total
        else:
            self.total = 1.

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            self.repeats += 1
            return 0.
        else:
            self.seen.add(tuple(self.action))
            energy = get_conformer_energies(self.molecule)[0]
            energy = energy * self.temp_0
            return np.exp(-1.0 * (energy - self.standard_energy)) / self.total

    def _get_info(self):
        info = super()._get_info()
        done = self._get_done()

        if (done):
            info['repeats'] = self.repeats
        
        return info

class UniqueGibbsRewardMixin(GibbsRewardMixin):
    def _get_reward(self):
        self.seen.add(tuple(self.action))
        current = get_conformer_energies(self.molecule)[0]
        current = current * self.temp_0

        rew = np.exp(-1.0 * (current - self.standard_energy)) / self.total

        if self._get_done():
            rew -= self._done_neg_reward()
        return rew

    def _done_neg_reward(self):
        before_total = np.exp(-1.0 * (get_conformer_energies(self.backup_mol) - self.standard_energy)).sum()
        before_conformers = self.backup_mol.GetNumConformers()
        self.backup_mol = prune_conformers(self.backup_mol, self.pruning_thresh)
        after_total = np.exp(-1.0 * (get_conformer_energies(self.backup_mol) - self.standard_energy)).sum()
        after_conformers = self.backup_mol.GetNumConformers()
        diff = before_total - after_total
        return diff / self.total

    def _update_memory(self):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.molecule)
            return

        c = self.molecule.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)

class PruningGibbsRewardMixin(GibbsRewardMixin):
    def _parse_molecule(self):
        super()._parse_molecule()
        
        mol = self.selected_config
        self.pruning_thresh = mol.pruning_thresh

        self.total_energy = 0
        self.backup_mol = Chem.Mol(self.molecule)
        self.backup_energys = list(get_conformer_energies(self.backup_mol))
        

    def _get_reward(self):
        self.seen.add(tuple(self.action))
        current = get_conformer_energies(self.molecule)[0]
        current = current * self.temp_0

        rew = np.exp(-1.0 * (current - self.standard_energy)) / self.total

        if self.current_step > 1:
            rew -= self._done_neg_reward(current)

        if self._get_done():
            self.backup_energys = []

        return rew

    def _done_neg_reward(self, current_energy):
        before_total = np.exp(-1.0 * (np.array(self.backup_energys) - self.standard_energy)).sum()

        self.backup_mol, energy_args = prune_last_conformer(self.backup_mol, self.pruning_thresh, self.backup_energys)
        self.backup_energys = list(np.array(self.backup_energys)[np.array(energy_args)])

        after_total = np.exp(-1.0 * (np.array(self.backup_energys) - self.standard_energy)).sum()

        assert self.backup_mol.GetNumConformers() == len(self.backup_energys)

        diff = before_total - after_total
        return diff / self.total

    def _update_memory(self):
        c = self.molecule.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)
        self.backup_energys += list(get_conformer_energies(self.molecule))

class LogPruningGibbsRewardMixin(PruningGibbsRewardMixin):
    def _get_reward(self):
        self.seen.add(tuple(self.action))

        if self.current_step > 1:
            self.done_neg_reward()

        energys = np.array(self.backup_energys) * self.temp_normal

        now = np.log(np.sum(np.exp(-1.0 * (np.array(energys) - self.standard_energy)) / self.total))
        if not np.isfinite(now):
            logging.error('neg inf reward')
            now = np.finfo(np.float32).eps
        rew = now - self.episode_reward

        if self._get_done():
            self.backup_energys = []

        return rew

    def done_neg_reward(self):
        before_total = np.exp(-1.0 * (np.array(self.backup_energys) - self.standard_energy)).sum()

        self.backup_mol, energy_args = prune_last_conformer(self.backup_mol, self.pruning_thresh, self.backup_energys)
        self.backup_energys = list(np.array(self.backup_energys)[np.array(energy_args)])

        assert self.backup_mol.GetNumConformers() == len(self.backup_energys)