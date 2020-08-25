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
        super().__init__()

        self.gibbs_normalize = gibbs_normalize
        self.eval = eval
        self.in_order = in_order
        self.temp_normal = temp_normal
        self.pruning_thresh = pruning_thresh
        self.all_files = glob.glob(f'{folder_name}*.json')
        self.folder_name = folder_name

        if sort_by_size:
            self.all_files.sort(key=os.path.getsize)
        else:
            self.all_files.sort()
        
        self.choice = -1
        self.episode_reward = 0
        self.choice_ind = 1
        self.num_good_episodes = 0

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

        self.everseen = set()
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]
        logging.info(f'rbn: {len(self.nonring)}')

        self.delta_t = []
        self.current_step = 0
        self.seen = set()
        self.energys = []
        self.zero_steps = 0
        self.repeats = 0

    def load(self, obj):
        pass

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
        done = (self.current_step == 200)
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
        self.zero_steps = 0
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

class Diff(SetGibbs):
    def __init__(self):
        super(Diff, self).__init__('molecules/diff/')

