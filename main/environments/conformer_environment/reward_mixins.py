import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints

from .conformer_env import ConformerEnv, confgen
from ...utils import ConformerGeneratorCustom, print_torsions, prune_conformers, prune_last_conformer, prune_last_conformer_quick

class UniqueRewardMixin(ConformerEnv):
    def _get_reward(self):
        self.seen.add(tuple(self.action))
        current = confgen.get_conformer_energies(self.molecule)[0]
        current = current * self.temp_0

        rew = np.exp(-1.0 * (current - self.standard_energy)) / self.total

        if self._get_done():
            rew -= self._done_neg_reward()
        return rew

    def _done_neg_reward(self):
        before_total = np.exp(-1.0 * (confgen.get_conformer_energies(self.backup_mol) - self.standard_energy)).sum()
        before_conformers = self.backup_mol.GetNumConformers()
        self.backup_mol = prune_conformers(self.backup_mol, self.pruning_thresh)
        after_total = np.exp(-1.0 * (confgen.get_conformer_energies(self.backup_mol) - self.standard_energy)).sum()
        after_conformers = self.backup_mol.GetNumConformers()
        diff = before_total - after_total
        return diff / self.total

    def _update_memory(self):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.molecule)
            return

        c = self.molecule.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)

class PruningRewardMixin(ConformerEnv):
    def _get_reward(self):
        self.seen.add(tuple(self.action))
        current = confgen.get_conformer_energies(self.molecule)[0]
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
        if self.current_step == 1:
            self.total_energy = 0
            self.backup_mol = Chem.Mol(self.molecule)
            self.backup_energys = list(confgen.get_conformer_energies(self.backup_mol))
            return

        c = self.molecule.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)
        self.backup_energys += list(confgen.get_conformer_energies(self.molecule))