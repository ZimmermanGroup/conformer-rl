import logging
from typing import List
import copy

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import TorsionFingerprints
import gym

from conformer_rl.config import MolConfig
from conformer_rl.environments.conformer_env import ConformerEnv

class CurriculumConformerEnv(ConformerEnv):

    def __init__(self, mol_configs: List[MolConfig]):
        gym.Env.__init__(self)
        logging.debug('initializing curriculum conformer environment')
        self.configs = copy.deepcopy(mol_configs)
        self.curriculum_max_index = 1

        self.config = self.configs[0]
        self.mol = self.config.mol
        self.mol.RemoveAllConformers()
        if Chem.EmbedMolecule(self.mol, randomSeed=self.config.seed, useRandomCoords=True) == -1:
            raise Exception('Unable to embed molecule with conformer using rdkit')
        self.conf = self.mol.GetConformer()
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]

        self.reset()

    def reset(self):
        logging.debug('reset called')

        self.total_reward = 0
        self.current_step = 0
        self.step_info = {}
        self.episode_info = {}

        # set index for the next molecule based on curriculum
        if self.curriculum_max_index == 1:
            index = 0
        else:
            p = 0.5 * np.ones(self.curriculum_max_index) / (self.curriculum_max_index - 1)
            p[-1] = 0.5
            index = np.random.choice(self.curriculum_max_index, p=p)

        logging.debug(f'Current Curriculum Molecule Index: {index}')

        # set up current molecule
        mol_config = self.configs[index]
        self.config = mol_config
        self.max_steps = mol_config.num_conformers
        self.mol = mol_config.mol
        self.mol.RemoveAllConformers()
        if Chem.EmbedMolecule(self.mol, randomSeed=self.config.seed, useRandomCoords=True) == -1:
            raise Exception('Unable to embed molecule with conformer using rdkit')
        self.conf = self.mol.GetConformer()
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]

        self.episode_info['mol'] = Chem.Mol(self.mol)
        self.episode_info['mol'].RemoveAllConformers()

        obs = self._obs()
        return obs


    def increase_level(self):
        self.curriculum_max_index = min(self.curriculum_max_index * 2, len(self.configs))

    def decrease_level(self):
        if self.curriculum_max_index > 1:
            self.curriculum_max_index = self.curriculum_max_index // 2