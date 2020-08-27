import time
import os.path
import multiprocessing
import logging
import glob
import json
import gym

import numpy as np
import pandas as pd
import scipy

from rdkit import Chem
from rdkit.Chem import AllChem

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance, NormalizeScale, Center, NormalizeRotation

from utils.moleculeToVector import *
from environment.graphenvironments import *

class RandomEndingSetGibbs(SetGibbs):
    @property
    def done(self):
        if self.current_step == 1:
            self.max_steps = np.random.randint(30, 50) * 5

        done = (self.current_step == self.max_steps)
        return done

class LongEndingSetGibbs(SetGibbs):
    @property
    def done(self):
        self.max_steps = 1000
        done = (self.current_step == self.max_steps)
        return done

class SetEnergy(SetGibbs):
    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
            return 0.0
        else:
            self.seen.add(tuple(self.action))
            print('standard', self.standard_energy)
            current = confgen.get_conformer_energies(self.mol)[0] * self.temp_normal
            print('current', current )
            if current - self.standard_energy > 20.0:
                return 0.0
            return self.standard_energy / (20 * current)

class SetEval(SetGibbs):
    def mol_appends(self):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.mol)
            return

        c = self.mol.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)

        if self.done:
            import pickle
            with open('test_mol.pickle', 'wb') as fp:
                pickle.dump(self.backup_mol, fp)


class SetEvalNoPrune(SetEval):
    def _get_reward(self):
        current = confgen.get_conformer_energies(self.mol)[0]
        current = current * self.temp_normal
        print('standard', self.standard_energy)
        print('current', current)

        rew = np.exp(-1.0 * (current - self.standard_energy)) / self.total
        return rew

class UniqueSetGibbs(SetGibbs):
    def _get_reward(self):
        self.seen.add(tuple(self.action))
        current = confgen.get_conformer_energies(self.mol)[0]
        current = current * self.temp_normal
        print('standard', self.standard_energy)
        print('current', current)

        rew = np.exp(-1.0 * (current - self.standard_energy)) / self.total

        done = (self.current_step == 200)
        if done:
            rew -= self.done_neg_reward()
        return rew

    def done_neg_reward(self):
        before_total = np.exp(-1.0 * (confgen.get_conformer_energies(self.backup_mol) - self.standard_energy)).sum()
        self.backup_mol = prune_conformers(self.backup_mol, 0.05)
        after_total = np.exp(-1.0 * (confgen.get_conformer_energies(self.backup_mol) - self.standard_energy)).sum()

        diff = before_total - after_total
        print('diff is ', diff)
        return diff / self.total

    def mol_appends(self):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.mol)
            return

        c = self.mol.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)
        if self.done and self.eval:
            import pickle
            i = 0
            while True:
                if os.path.exists(f'test_mol{i}.pickle'):
                    i += 1
                    continue
                else:
                    with open(f'test_mol{i}.pickle', 'wb') as fp:
                        pickle.dump(self.backup_mol, fp)
                    break

class PruningSetGibbsQuick(SetGibbs):
    def _get_reward(self):
        self.seen.add(tuple(self.action))
        current = confgen.get_conformer_energies(self.mol)[0]
        current = current * self.temp_normal
        print('standard', self.standard_energy)
        print('current', current)

        rew = np.exp(-1.0 * (current - self.standard_energy)) / self.total
        print('current step', self.current_step)
        if self.current_step > 1:
            #figure out keep or not keep
            rew *= self.done_neg_reward()

        return rew

    def done_neg_reward(self):
        self.backup_mol, ret_cond = prune_last_conformer_quick(self.backup_mol, self.pruning_thresh)
        return ret_cond

    def mol_appends(self):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.mol)
            return

        c = self.mol.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)

class PruningSetEnergyQuick(SetEnergy):
    def _get_reward(self):
        self.seen.add(tuple(self.action))
        print('standard', self.standard_energy)
        print('current', current)

        if current - self.standard_energy > 20.0:
            rew = 0.0
        rew = self.standard_energy / (20 * current)

        if self.current_step > 1:
            #figure out keep or not keep
            rew *= self.done_neg_reward()

        return rew

    def done_neg_reward(self):
        self.backup_mol, ret_cond = prune_last_conformer_quick(self.backup_mol, self.pruning_thresh)
        return ret_cond

    def mol_appends(self):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.mol)
            return

        c = self.mol.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)

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

class SetCurricula(SetGibbs):
    def info(self, info):
        info['num_good_episodes'] = self.num_good_episodes
        info['choice_ind'] = self.choice_ind
        return info

    def molecule_choice(self):
        if self.episode_reward > 0.75:
            self.num_good_episodes += 1
        else:
            self.num_good_episodes = 0

        if self.num_good_episodes >= 5:
            self.choice_ind *= 2
            self.choice_ind = min(self.choice_ind, len(self.all_files))
            self.num_good_episodes = 0

        cjson = np.random.choice(self.all_files[0:self.choice_ind])

        print(cjson, '\n\n\n\n')

        with open(cjson) as fp:
            obj = json.load(fp)
        return obj

class SetCurriculaExp(SetGibbs):
    def info(self, info):
        info['num_good_episodes'] = self.num_good_episodes
        info['choice_ind'] = self.choice_ind
        return info

    def molecule_choice(self):
        if self.episode_reward > 1.20:
            self.num_good_episodes += 1
        else:
            self.num_good_episodes = 0

        if self.num_good_episodes >= 5:
            self.choice_ind += 1
            self.choice_ind = min(self.choice_ind, len(self.all_files))
            self.num_good_episodes = 0

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


class SetCurriculaForgetting(SetGibbs):
    def info(self, info):
        info['num_good_episodes'] = self.num_good_episodes
        info['choice_ind'] = self.choice_ind
        return info

    def molecule_choice(self):
        if self.episode_reward > 0.90:
            self.num_good_episodes += 1
        else:
            self.num_good_episodes = 0

        if self.num_good_episodes >= 10:
            self.choice_ind += 1
            self.choice_ind = min(self.choice_ind, len(self.all_files))
            self.num_good_episodes = 0

        cjson = self.all_files[self.choice_ind - 1]
        print(cjson, '\n\n\n\n')

        with open(cjson) as fp:
            obj = json.load(fp)
        return obj

class SetGibbsStupid(SetGibbs):
    def _get_obs(self):
        data = Batch.from_data_list([mol2vecstupidsimple(self.mol)])
        return data, self.nonring

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

class SetGibbsSkeletonPoints(SetGibbs):
    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeletonpoints(self.mol)])
        return data, self.nonring