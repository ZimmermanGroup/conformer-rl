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
import logging
import torch
import pandas as pd
import time

import torch

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance, NormalizeScale, Center, NormalizeRotation

import glob
import json
import logging

from ..utils import ConformerGeneratorCustom, print_torsions, prune_conformers, prune_last_conformer, prune_last_conformer_quick


confgen = ConformerGeneratorCustom(max_conformers=1,
                             rmsd_threshold=None,
                             force_field='mmff',
                             pool_multiplier=1)

class SetGibbs(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
            self, 
            folder_name, 
            gibbs_normalize=True, 
            eval=False, 
            in_order=False, 
            temp_normal=1.0, 
            sort_by_size=True,
            pruning_thresh=0.05,
        ):
        super(SetGibbs, self).__init__()
        self.gibbs_normalize = gibbs_normalize
        self.temp_normal = temp_normal
        self.all_files = glob.glob(f'{folder_name}*.json')
        self.folder_name = folder_name

        if sort_by_size:
            self.all_files.sort(key=os.path.getsize)
        else:
            self.all_files.sort()

        self.in_order = in_order
        self.eval = eval
        self.pruning_thresh = pruning_thresh
        
        self.choice = -1
        self.choice_ind = 1
        self.num_good_episodes = 0

        self.delta_t = []
        self.energys = []

        if '/' in self.folder_name:
            self.folder_name = self.folder_name.split('/')[0]

        self.reset()

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            self.repeats += 1
            return 0
        else:
            self.seen.add(tuple(self.action))
            current = confgen.get_conformer_energies(self.mol)[0]
            current = current * self.temp_normal
            return np.exp(-1.0 * (current - self.standard_energy)) / self.total

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeletonpoints(self.mol)])
        return data, self.nonring

    def step(self, action):
        # Execute one time step within the environment
        if len(action.shape) > 1:
            self.action = action[0]
        else:
            self.action = action
        self.current_step += 1

        desired_torsions = []

        for idx, tors in enumerate(self.nonring):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * self.action[idx]
            desired_torsions.append(ang)
            try:
                Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], float(ang))
            except:
                Chem.MolToMolFile(self.mol, 'debug.mol')
                logging.error('exit with debug.mol')
                exit(0)
        Chem.AllChem.MMFFOptimizeMolecule(self.mol, confId=0)

        self.mol_appends()

        obs = self._get_obs()
        rew = self._get_reward()
        self.episode_reward += rew

        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)

        info = {}
        if self.done:
            info['repeats'] = self.repeats

        info = self.info(info)

        return obs, rew, self.done, info

    @property
    def done(self):
        done = (self.current_step == 5)
        return done

    def info(self, info):
        return info

    def mol_appends(self):
        pass

    def molecule_choice(self):
        cjson = np.random.choice(self.all_files)
        with open(cjson) as fp:
            obj = json.load(fp)
        return obj

    def reset(self):
        self.repeats = 0
        self.current_step = 0
        self.seen = set()
        while True:
            obj = self.molecule_choice()

            if 'inv_temp' in obj:
                self.temp_normal = obj['inv_temp']

            self.standard_energy = float(obj['standard'])
            if 'total' in obj and self.gibbs_normalize:
                self.total = obj['total']
            else:
                self.total = 1.0

            if 'mol' in obj:
                self.mol = Chem.MolFromSmiles(obj['mol'])
                self.mol = Chem.AddHs(self.mol)
                res = AllChem.EmbedMultipleConfs(self.mol, numConfs=1)
                if not len(res):
                    continue
                res = Chem.AllChem.MMFFOptimizeMoleculeConfs(self.mol)
                self.conf = self.mol.GetConformer(id=0)
            else:
                self.mol = Chem.MolFromMolFile(os.path.join(self.folder_name, obj['molfile']))
                self.mol = Chem.AddHs(self.mol)
                self.conf = self.mol.GetConformer(id=0)
                res = Chem.AllChem.MMFFOptimizeMoleculeConfs(self.mol)
            break

        self.episode_reward = 0
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]
        logging.info(f'rbn: {len(self.nonring)}')

        obs = self._get_obs()


        print('step time mean', np.array(self.delta_t).mean())
        print('reset called\n\n\n\n\n')
        print_torsions(self.mol)
        return obs

    def render(self, mode='human', close=False):
        print_torsions(self.mol)

    def change_level(self, up_or_down):
        pass

class UniqueSetGibbs(SetGibbs):
    def _get_reward(self):
        self.seen.add(tuple(self.action))
        current = confgen.get_conformer_energies(self.mol)[0]
        current = current * self.temp_normal
        print('standard', self.standard_energy)
        print('current', current)

        rew = np.exp(-1.0 * (current - self.standard_energy)) / self.total

        if self.done:
            rew -= self.done_neg_reward()
        return rew

    def done_neg_reward(self):
        before_total = np.exp(-1.0 * (confgen.get_conformer_energies(self.backup_mol) - self.standard_energy)).sum()
        before_conformers = self.backup_mol.GetNumConformers()
        self.backup_mol = prune_conformers(self.backup_mol, self.pruning_thresh)
        after_total = np.exp(-1.0 * (confgen.get_conformer_energies(self.backup_mol) - self.standard_energy)).sum()
        after_conformers = self.backup_mol.GetNumConformers()
        diff = before_total - after_total
        print('diff is ', diff)
        print(f'pruned {after_conformers - before_conformers} conformers')
        print(f'pruning thresh is {self.pruning_thresh}')
        return diff / self.total

    def mol_appends(self):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.mol)
            return

        c = self.mol.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)

class PruningSetGibbs(SetGibbs):
    def _get_reward(self):
        self.seen.add(tuple(self.action))
        current = confgen.get_conformer_energies(self.mol)[0]
        current = current * self.temp_normal
        print('standard', self.standard_energy)
        print('current', current)

        rew = np.exp(-1.0 * (current - self.standard_energy)) / self.total

        print('current step', self.current_step)
        if self.current_step > 1:
            rew -= self.done_neg_reward(current)

        if self.done:
            self.backup_energys = []

        return rew

    def done_neg_reward(self, current_energy):
        before_total = np.exp(-1.0 * (np.array(self.backup_energys) - self.standard_energy)).sum()

        self.backup_mol, energy_args = prune_last_conformer(self.backup_mol, self.pruning_thresh, self.backup_energys)
        self.backup_energys = list(np.array(self.backup_energys)[np.array(energy_args)])

        after_total = np.exp(-1.0 * (np.array(self.backup_energys) - self.standard_energy)).sum()

        assert self.backup_mol.GetNumConformers() == len(self.backup_energys)

        diff = before_total - after_total
        return diff / self.total

    def mol_appends(self):
        if self.current_step == 1:
            self.total_energy = 0
            self.backup_mol = Chem.Mol(self.mol)
            self.backup_energys = list(confgen.get_conformer_energies(self.backup_mol))
            print('num_energys', len(self.backup_energys))
            return

        c = self.mol.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)
        self.backup_energys += list(confgen.get_conformer_energies(self.mol))
        print('num_energys', len(self.backup_energys))

class PruningSetLogGibbs(PruningSetGibbs):
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

        if self.done:
            self.backup_energys = []

        return rew

    def done_neg_reward(self):
        before_total = np.exp(-1.0 * (np.array(self.backup_energys) - self.standard_energy)).sum()

        self.backup_mol, energy_args = prune_last_conformer(self.backup_mol, self.pruning_thresh, self.backup_energys)
        self.backup_energys = list(np.array(self.backup_energys)[np.array(energy_args)])

        assert self.backup_mol.GetNumConformers() == len(self.backup_energys)

class SetCurriculaExtern(SetGibbs):
    def info(self, info):
        info['choice_ind'] = self.choice_ind
        return info

    def molecule_choice(self):
        if self.choice_ind != 1:
            p = 0.5 * np.ones(self.choice_ind) / (self.choice_ind - 1)
            p[-1] = 0.5
            cjson = np.random.choice(self.all_files[0:self.choice_ind], p=p)
        else:
            cjson = self.all_files[0]

        print(cjson, '\n\n\n\n')

        with open(cjson) as fp:
            obj = json.load(fp)
        return obj

    def change_level(self, up_or_down):
        if up_or_down:
            self.choice_ind *= 2

        else:
            if self.choice_ind != 1:
                self.choice_ind = int(self.choice_ind / 2)

        self.choice_ind = min(self.choice_ind, len(self.all_files))

class SetGibbsDense(SetGibbs):
    def _get_obs(self):
        data = Batch.from_data_list([mol2vecdense(self.mol)])
        return data, self.nonring

class SetGibbsSkeleton(SetGibbs):
    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

class SetGibbsSkeletonFeatures(SetGibbs):
    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class Diff(SetGibbs):
    def __init__(self):
        super(Diff, self).__init__('diff/')

class TenTorsionSetCurriculumPoints(SetCurriculaExtern, PruningSetGibbs):
    def __init__(self):
        super(TenTorsionSetCurriculumPoints, self).__init__('huge_hc_set/10_')        

class AlkaneValidation10(UniqueSetGibbs):
    def __init__(self):
        super(AlkaneValidation10, self).__init__('alkane_validation_10/')

class AlkaneTest11(UniqueSetGibbs):
    def __init__(self):
        super(AlkaneTest11, self).__init__('alkane_test_11/')

class AlkaneTest22(UniqueSetGibbs):
    def __init__(self):
        super(AlkaneTest22, self).__init__('alkane_test_22/')

class LigninAllSetPruningLogSkeletonCurriculumLong(SetCurriculaExtern, PruningSetLogGibbs, LongEndingSetGibbs):
    def __init__(self):
        super(LigninAllSetPruningLogSkeletonCurriculumLong, self).__init__('lignin_hightemp/', temp_normal=0.25, sort_by_size=False)

class LigninAllSetPruningLogSkeletonCurriculumLong015(SetCurriculaExtern, PruningSetLogGibbs, LongEndingSetGibbs):
    def __init__(self):
        super(LigninAllSetPruningLogSkeletonCurriculumLong015, self).__init__('lignin_hightemp/', temp_normal=0.25, sort_by_size=False, pruning_thresh=0.15)

class LigninPruningSkeletonValidationLong(UniqueSetGibbs, LongEndingSetGibbs):
    def __init__(self):
        super(LigninPruningSkeletonValidationLong, self).__init__('lignin_eval_sample/', temp_normal=0.25, sort_by_size=False)

## sgld normed 
class LigninPruningSkeletonEvalSgldFinalLong015(UniqueSetGibbs, LongEndingSetGibbs):
    def __init__(self):
        super(LigninPruningSkeletonEvalSgldFinalLong015, self).__init__('lignin_eval_final_sgld/', eval=False, temp_normal=0.2519, sort_by_size=False, pruning_thresh=0.15)

class LigninPruningSkeletonEvalSgldFinalLong015Save(UniqueSetGibbs, LongEndingSetGibbs):
    def __init__(self):
        super(LigninPruningSkeletonEvalSgldFinalLong015Save, self).__init__('lignin_eval_final_sgld/', eval=True, temp_normal=0.2519, sort_by_size=False, pruning_thresh=0.15)
