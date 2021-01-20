import numpy as np
import scipy
import gym

from rdkit import Chem, DataStructs, RDConfig, rdBase
from rdkit import rdBase
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.Chem import Draw,PyMol,rdFMCS
from rdkit.Chem.Draw import IPythonConsole

import os.path
import multiprocessing

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance, NormalizeScale, Center, NormalizeRotation

import logging

from ..utils import ConformerGeneratorCustom, print_torsions, prune_conformers, prune_last_conformer, prune_last_conformer_quick
from .molecule_handler import mol2vecskeletonpoints

confgen = ConformerGeneratorCustom(max_conformers=1,
                             rmsd_threshold=None,
                             force_field='mmff',
                             pool_multiplier=1)

class ConformerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mol_config, max_steps=5, temp_0=1.):
        super(ConformerEnv, self).__init__()

        self.temp_0 = temp_0 # temperature normalization constant
        self.mol_config = mol_config
        self.max_steps = max_steps
        self.total_reward = 0

        self.current_step = 0
        self.repeats = 0
        self.seen = set()

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

        return obs, reward, done, info
        
    def reset(self):
        self.total_reward = 0
        self.current_step = 0
        self.repeats = 0
        self.seen = set()

        self.mol = self._get_mol()

        self.molecule = self.mol.molecule

        if self.mol.inv_temp is not None:
            self.temp_0 = self.mol.inv_temp

        self.standard_energy = self.mol.standard

        if self.mol.total is not None:
            self.total = self.mol.total
        else:
            self.total = 1.

        self.conf = self.molecule.GetConformer(id=0)
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.molecule)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]

        obs = self._get_obs()
        return obs

    def render(self, mode='human'):
        pass

    def _get_mol(self):
        return self.mol_config[0]

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeletonpoints(self.molecule)])
        return data, self.nonring

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            self.repeats += 1
            return 0.
        else:
            self.seen.add(tuple(self.action))
            energy = confgen.get_conformer_energies(self.molecule)[0]
            energy = energy * self.temp_0
            return np.exp(-1.0 * (energy - self.standard_energy)) / self.total

    def _get_done(self):
        return (self.current_step == self.max_steps)

    def _get_info(self):
        info = {}
        done = self._get_done()
        if (done):
            info['repeats'] = self.repeats
            info['episodic_return'] = self.total_reward
        else:
            info['episodic_return'] = None
        
        return info

    def _handle_action(self):
        desired_torsions = []

        for idx, tors in enumerate(self.nonring):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * self.action[idx]
            desired_torsions.append(ang)
            Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], float(ang))
        Chem.AllChem.MMFFOptimizeMolecule(self.molecule, confId=0)

    def _update_memory(self):
        pass


        