from utils import *

import numpy as np
import scipy

from rdkit import Chem
from rdkit.Chem import AllChem

import os.path
import multiprocessing
import logging
import torch
import pandas as pd
import time

import torch

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance, NormalizeScale

import glob
import json

def bond_features(bond, use_chirality=False, use_basic_feats=True):
    bt = bond.GetBondType()
    bond_feats = []
    if use_basic_feats:
        bond_feats = bond_feats + [
            bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)

def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def atom_features(atom, conf):

    anum = atom.GetSymbol()
    atom_feats = []

    atom_feats = atom_feats + [
        anum == 'C', anum == 'O',
    ]

    fts = np.array(atom_feats)

    p = conf.GetAtomPosition(atom.GetIdx())

    return fts, np.array([p.x, p.y, p.z])


def atom_features_simple(atom, conf):
    p = conf.GetAtomPosition(atom.GetIdx())
    return np.array([p.x, p.y, p.z])

def mol2vecsimple(mol):
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [atom_features_simple(atom, conf) for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False) for bond in bonds]
    for bond in bonds:
        edge_attr.append(bond_features(bond))
    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )
    data = Distance()(data)
    return data

def mol2vecstupidsimple(mol):
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [[] for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=False) for bond in bonds]
    for bond in bonds:
        edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=False))
    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    data = NormalizeScale()(data)
    data = Distance(norm=False)(data)
    data.x = data.pos

    e = data.edge_attr
    new_e = -1 + ((e - e.min())*2)/(e.max() - e.min())
    data.edge_attr = new_e

    return data

def mol2vecskeleton(mol):
    mol = Chem.rdmolops.RemoveHs(mol)
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f = [[] for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=False) for bond in bonds]

    for bond in bonds:
        edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=False))

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    data = NormalizeScale()(data)
    data = Distance(norm=False)(data)
    data.x = data.pos

    return data

def mol2vecskeleton_features(mol):
    mol = Chem.rdmolops.RemoveHs(mol)
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [[] for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=True) for bond in bonds]
    for bond in bonds:
        edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=True))

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    data = NormalizeScale()(data)
    data = Distance(norm=False)(data)
    data.x = data.pos

    return data

def mol2vecdense(mol):
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    n = len(atoms)

    edge_index = []
    edge_attr = []


    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edge_index.append([i, j])
            edge_attr.append(adj[i][j])


    node_f= [[] for atom in atoms]

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).T,
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    data = NormalizeScale()(data)
    data = Distance(norm=False)(data)
    data.x = data.pos

    return data

def mol2vecbasic(mol):
    mol = Chem.rdmolops.RemoveHs(mol)
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [[] for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False, use_basic_feats=False) for bond in bonds]
    for bond in bonds:
        edge_attr.append(bond_features(bond, use_chirality=False, use_basic_feats=False))

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )

    data = NormalizeScale()(data)
    data = Distance(norm=False)(data)
    data.x = data.pos

    e = data.edge_attr
    new_e = -1 + ((e - e.min())*2)/(e.max() - e.min())
    data.edge_attr = new_e

    return data


# def mol2points(mol):
#     conf = mol.GetConformer(id=-1)
#     atoms = mol.GetAtoms()
#     bonds = mol.GetBonds()
#     node_f= [atom_features_simple(atom, conf) for atom in atoms]

#     data = Data(
#                 x=torch.tensor(node_f, dtype=torch.float),
#             )
#     data = Distance()(data)
#     return data

def sort_func(x, y):
        if x < y:
            return -1
        elif y < x:
            return 1
        else:
            if os.path.getsize(x) < os.path.getsize(y):
                return -1
            elif os.path.getsize(y) < os.path.getsize(x):
                return 1
            else:
                return 0


confgen = ConformerGeneratorCustom(max_conformers=1,
                             rmsd_threshold=None,
                             force_field='mmff',
                             pool_multiplier=1)

class BestGibbs(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, folder_name, gibbs_normalize=False, temp_normal=1.0, sort_by_size=True, ind_select=None):
        super(BestGibbs, self).__init__()
        self.temp_normal = temp_normal
        self.gibbs_normalize = gibbs_normalize
        self.all_files = glob.glob(f'{folder_name}*.json')
        self.folder_name = folder_name

        self.ind_select = ind_select

        if sort_by_size:
            self.all_files.sort(key=os.path.getsize)
        else:
            self.all_files.sort()

        self.choice = -1
        self.episode_reward = 0
        self.choice_ind = 1
        self.num_good_episodes = 0

        if '/' in self.folder_name:
            self.folder_name = self.folder_name.split('/')[0]

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

        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]
        self.delta_t = []
        self.current_step = 0
        self.best_seen = 999.9999
        self.energys = []
        self.zero_steps = 0
        self.repeats = 0

    def load(self, obj):
        pass

    def _get_reward(self):
        current = confgen.get_conformer_energies(self.mol)[0]
        current = current * self.temp_normal

        if current >= self.best_seen:
            print('seen better')
        else:
            self.best_seen = current
            print('current', self.best_seen)

        return np.exp(-1.0 * (self.best_seen - self.standard_energy)) / 20.0

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecstupidsimple(self.mol)])
        return data, self.nonring

    def step(self, action):
        # Execute one time step within the environment
        print("action is ", action)
        if len(action.shape) > 1:
            self.action = action[0]
        else:
            self.action = action
        self.current_step += 1

        begin_step = time.process_time()
        desired_torsions = []

        for idx, tors in enumerate(self.nonring):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * self.action[idx]
            desired_torsions.append(ang)
            try:
                Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], float(ang))
            except:
                Chem.MolToMolFile(self.mol, 'debug.mol')
                print('exit with debug.mol')
                exit(0)

        Chem.AllChem.MMFFOptimizeMolecule(self.mol, confId=0)

        rbn = len(self.nonring)
        if rbn == 3:
            done = (self.current_step == 25)
        else:
            done = (self.current_step == 200)

        self.mol_appends(done)

        obs = self._get_obs()
        rew = self._get_reward()
        self.episode_reward += rew

        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)

        end_step = time.process_time()

        delta_t = end_step-begin_step
        self.delta_t.append(delta_t)

        info = {}
        if done:
            info['repeats'] = self.repeats

        info = self.info(info)

        return obs, rew, done, info

    def info(self, info):
        return info

    def mol_appends(self, done):
        pass

    def change_level(self, up_or_down=True):
        print('level', up_or_down)

    def molecule_choice(self):
        if self.ind_select is not None:
            cjson = self.all_files[self.ind_select]
        else:
            cjson = np.random.choice(self.all_files)

        with open(cjson) as fp:
            obj = json.load(fp)
        return obj

    def reset(self):
        self.best_seen = 999.9999
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

        obs = self._get_obs()

        print('step time mean', np.array(self.delta_t).mean())
        print('reset called\n\n\n\n\n')
        print_torsions(self.mol)
        return obs

    def render(self, mode='human', close=False):
        print_torsions(self.mol)

class BestTestGibbs(BestGibbs):
    def __init__(self, **kwargs):
        super(BestTestGibbs, self).__init__(**kwargs)

    def _get_reward(self):
        current = confgen.get_conformer_energies(self.mol)[0]
        current = current * self.temp_normal

        mol = Chem.MolFromMolFile(f'{self.folder_name}{self.ind_select}.mol')

        if current <= self.best_seen:
            print('seen better')
        else:
            self.best_seen = current
            print('current', self.best_seen)

        return np.exp(-1.0 * (self.best_seen - self.standard_energy)) / 20.0

class TestBestGibbs(BestGibbs):
    def __init__(self):
        super(TestBestGibbs, self).__init__('diff/')

class SetGibbs(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, folder_name, gibbs_normalize=True, in_order=False, temp_normal=1.0, sort_by_size=True):
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
        self.choice = -1
        self.episode_reward = 0
        self.choice_ind = 1
        self.num_good_episodes = 0

        if '/' in self.folder_name:
            self.folder_name = self.folder_name.split('/')[0]

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
            print('already seen')
            self.repeats += 1
            return 0
        else:
            self.seen.add(tuple(self.action))
            current = confgen.get_conformer_energies(self.mol)[0]
            current = current * self.temp_normal
            print('standard', self.standard_energy)
            print('current', current)
            return np.exp(-1.0 * (current - self.standard_energy)) / self.total

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

    def step(self, action):
        # Execute one time step within the environment
        print("action is ", action)
        if len(action.shape) > 1:
            self.action = action[0]
        else:
            self.action = action
        self.current_step += 1

        begin_step = time.process_time()
        desired_torsions = []

        for idx, tors in enumerate(self.nonring):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * self.action[idx]
            desired_torsions.append(ang)
            try:
                Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], float(ang))
            except:
                Chem.MolToMolFile(self.mol, 'debug.mol')
                print('exit with debug.mol')
                exit(0)
        Chem.AllChem.MMFFOptimizeMolecule(self.mol, confId=0)

        rbn = len(self.nonring)
        if rbn == 3:
            done = (self.current_step == 25)
        else:
            done = (self.current_step == 200)

        self.mol_appends(done)

        obs = self._get_obs()
        rew = self._get_reward()
        self.episode_reward += rew

        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)

        end_step = time.process_time()

        delta_t = end_step-begin_step
        self.delta_t.append(delta_t)

        info = {}
        if done:
            info['repeats'] = self.repeats

        info = self.info(info)

        return obs, rew, done, info

    def info(self, info):
        return info

    def mol_appends(self, done):
        pass

    def molecule_choice(self):
        if self.in_order:
            self.choice = (self.choice + 1) % len(self.all_files)
            cjson = self.all_files[self.choice]
        else:
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

        obs = self._get_obs()

        print('step time mean', np.array(self.delta_t).mean())
        print('reset called\n\n\n\n\n')
        print_torsions(self.mol)
        return obs

    def render(self, mode='human', close=False):
        print_torsions(self.mol)

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
    def mol_appends(self, done):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.mol)
            return

        if done:
            import pickle
            with open('test_mol.pickle', 'wb') as fp:
                pickle.dump(self.backup_mol, fp)
        else:
            c = self.mol.GetConformer(id=0)
            self.backup_mol.AddConformer(c, assignId=True)


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

    def mol_appends(self, done):
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
            rew -= self.done_neg_reward()

        if self.current_step == 200:
            self.backup_energys = []

        return rew

    def done_neg_reward(self):
        before_total = np.exp(-1.0 * (confgen.get_conformer_energies(self.backup_mol) - self.standard_energy)).sum()

        self.backup_mol, energy_args = prune_last_conformer(self.backup_mol, 0.05, self.backup_energys)
        print(energy_args)
        after_total = np.exp(-1.0 * (confgen.get_conformer_energies(self.backup_mol) - self.standard_energy)).sum()

        self.backup_energys = list(np.array(self.backup_energys)[np.array(energy_args)])

        assert self.backup_mol.GetNumConformers() == len(self.backup_energys)

        diff = before_total - after_total
        return diff / self.total

    def mol_appends(self, done):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.mol)
            self.backup_energys = list(confgen.get_conformer_energies(self.backup_mol))
            print('num_energys', len(self.backup_energys))
            return

        c = self.mol.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)
        self.backup_energys += list(confgen.get_conformer_energies(self.mol))
        print('num_energys', len(self.backup_energys))

class TrihexylEval(SetEval):
    def __init__(self):
        super(TrihexylEval, self).__init__('trihexyl/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecstupidsimple(self.mol)])
        return data, self.nonring

class TrihexylUnique(UniqueSetGibbs):
    def __init__(self):
        super(TrihexylUnique, self).__init__('trihexyl/')


class SetCurriculaLevels(SetGibbs):
    def info(self, info):
        self.choice_ind = min(self.choice_ind, len(self.all_files))
        info['choice_ind'] = self.choice_ind
        return info

    def molecule_choice(self):
        self.choice_ind = min(self.choice_ind, len(self.all_files))

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

    def change_level(up_or_down):
        if up_or_down:
            self.choice_ind += 1

        elif self.choice_ind != 1:
            self.choice_ind -= 1

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

class BestCurriculaExp(BestGibbs):
    def info(self, info):
        info['num_good_episodes'] = self.num_good_episodes
        info['choice_ind'] = self.choice_ind
        return info

    def molecule_choice(self):
        if self.episode_reward > 7.5:
            self.num_good_episodes += 1
        else:
            self.num_good_episodes = 0

        if self.num_good_episodes >= 10:
            filename = f'{self.choice_ind}'
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
    # def info(self, info):
    #     info['choice_ind'] = self.choice_ind
    #     return info

    # def molecule_choice(self):
    #     self.choice_ind = min(self.choice_ind, len(self.all_files))

    #     if self.choice_ind != 1:
    #         p = 0.5 * np.ones(self.choice_ind) / (self.choice_ind - 1)
    #         p[-1] = 0.5
    #         cjson = np.random.choice(self.all_files[0:self.choice_ind], p=p)
    #     else:
    #         cjson = self.all_files[0]

    #     print(cjson, '\n\n\n\n')

    #     with open(cjson) as fp:
    #         obj = json.load(fp)
    #     return obj

    # def change_level(up_or_down):
    #     if up_or_down:
    #         self.choice_ind += 1

    #     elif self.choice_ind != 1:
    #         self.choice_ind -= 1

class TChainTrain(BestCurriculaExp):
    def __init__(self):
        super(TChainTrain, self).__init__('transfer_test_t_chain/')

class TChainTest(BestTestGibbs):
    def __init__(self, **kwargs):
        super(TChainTest, self).__init__('transfer_test_t_chain/', **kwargs)

class SetEnergyCurriculaExp(SetEnergy):
    def info(self, info):
        info['num_good_episodes'] = self.num_good_episodes
        info['choice_ind'] = self.choice_ind
        return info

    def molecule_choice(self):
        if self.episode_reward > 6.0:
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

class StraightChainTen(SetGibbs):
    def __init__(self):
        super(StraightChainTen, self).__init__('straight_chain_10/')

class StraightChainTenEval(SetEval):
    def __init__(self):
        super(StraightChainTenEval, self).__init__('straight_chain_10/')

class StraightChainTenEleven(SetGibbs):
    def __init__(self):
        super(StraightChainTenEleven, self).__init__('straight_chain_10_11/')

class StraightChainElevenEval(SetEval):
    def __init__(self):
        super(StraightChainElevenEval, self).__init__('straight_chain_11/')

class StraightChainTenElevenTwelve(SetGibbs):
    def __init__(self):
        super(StraightChainTenElevenTwelve, self).__init__('straight_chain_10_11_12/')

class StraightChainTwelveEval(SetEval):
    def __init__(self):
        super(StraightChainTwelveEval, self).__init__('straight_chain_12/')

class AllThreeTorsionSet(SetGibbs):
    def __init__(self):
        super(AllThreeTorsionSet, self).__init__('huge_hc_set/3_')

class AllFiveTorsionSet(SetGibbs):
    def __init__(self):
        super(AllFiveTorsionSet, self).__init__('huge_hc_set/5_')

class AllEightTorsionSet(SetGibbs):
    def __init__(self):
        super(AllEightTorsionSet, self).__init__('huge_hc_set/8_')


class AllEightTorsionSetStupid(SetGibbsStupid):
    def __init__(self):
        super(AllEightTorsionSetStupid, self).__init__('huge_hc_set/8_')

class AllEightTorsionSetDense(SetGibbsDense):
    def __init__(self):
        super(AllEightTorsionSetDense, self).__init__('huge_hc_set/8_')

class AllTenTorsionSet(SetGibbs):
    def __init__(self):
        super(AllTenTorsionSet, self).__init__('huge_hc_set/10_')

class AllTenTorsionSetPruning(PruningSetGibbs):
    def __init__(self):
        super(AllTenTorsionSetPruning, self).__init__('huge_hc_set/10_')

class TenTorsionSetCurriculumPruning(PruningSetGibbs, SetCurricula):
    def __init__(self):
        super(TenTorsionSetCurriculumPruning, self).__init__('huge_hc_set/10_')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

class TenTorsionSetCurriculum(SetCurricula):
    def __init__(self):
        super(TenTorsionSetCurriculum, self).__init__('huge_hc_set/10_')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

class TenTorsionSetCurriculumExp(SetCurriculaExp):
    def __init__(self):
        super(TenTorsionSetCurriculumExp, self).__init__('huge_hc_set/10_')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

class NewLigninCurr(SetCurriculaExp):
    def __init__(self):
        super(NewLigninCurr, self).__init__('lignin_ob_hightemp_h_fixed/', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class NewLigninEval(SetGibbs):
    def __init__(self):
        super(NewLigninEval, self).__init__('lignin_ob_hightemp_h_fixed_eval/', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class NewEnergyLigninCurr(SetEnergyCurriculaExp):
    def __init__(self):
        super(NewEnergyLigninCurr, self).__init__('lignin_ob_hightemp_h_fixed/', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class NewEnergyLigninEval(SetEnergy):
    def __init__(self):
        super(NewEnergyLigninEval, self).__init__('lignin_ob_hightemp_h_fixed_eval/', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class TenTorsionSetCurriculumForgetting(SetCurriculaForgetting):
    def __init__(self):
        super(TenTorsionSetCurriculumForgetting, self).__init__('huge_hc_set/10_')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

class DifferentCarbonSet(SetEval):
    def __init__(self):
        super(DifferentCarbonSet, self).__init__('diff/')

class DifferentCarbonSet11(SetEval):
    def __init__(self):
        super(DifferentCarbonSet11, self).__init__('diff_11/')

class Trihexyl(SetGibbs):
    def __init__(self):
        super(Trihexyl, self).__init__('trihexyl/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecbasic(self.mol)])
        return data, self.nonring


class GiantSet(SetGibbs):
    def __init__(self):
        super(GiantSet, self).__init__('giant_hc_set/')

class OneSet(SetGibbs):
    def __init__(self):
        super(OneSet, self).__init__('one_set/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

class TwoSet(SetGibbs):
    def __init__(self):
        super(TwoSet, self).__init__('two_set/')

class ThreeSet(SetGibbs):
    def __init__(self):
        super(ThreeSet, self).__init__('three_set/')

class ThreeSetPruning(PruningSetGibbs):
    def __init__(self):
        super(ThreeSetPruning, self).__init__('three_set/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

class FourSet(SetGibbs):
    def __init__(self):
        super(FourSet, self).__init__('four_set/')

class DifferentCarbonSetUnique(UniqueSetGibbs):
    def __init__(self):
        super(DifferentCarbonSetUnique, self).__init__('diff/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecstupidsimple(self.mol)])
        return data, self.nonring

class FourSetUnique(UniqueSetGibbs):
    def __init__(self):
        super(FourSetUnique, self).__init__('four_set/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecstupidsimple(self.mol)])
        return data, self.nonring

class OneSetUnique(UniqueSetGibbs):
    def __init__(self):
        super(OneSetUnique, self).__init__('one_set/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

class OneSetPruning(PruningSetGibbs):
    def __init__(self):
        super(OneSetPruning, self).__init__('one_set/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

class LigninAllSet(SetGibbs):
    def __init__(self):
        super(LigninAllSet, self).__init__('lignins_out/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring


class LigninAllSet2(SetGibbs):
    def __init__(self):
        super(LigninAllSet2, self).__init__('lignins_out/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
            return 0.0
        else:
            self.seen.add(tuple(self.action))
            current = confgen.get_conformer_energies(self.mol)[0] * 0.5

            print('standard', self.standard_energy)
            print('current', current)

            if current - self.standard_energy > 10.0:
                return 0.0

            x = current - self.standard_energy
            return 1.0 - x/10.0

class LigninTwoLowTempEval(SetEval):
    def __init__(self):
        super(LigninTwoLowTempEval, self).__init__('lignins_out_high_temp/2_', temp_normal=0.25)

    # def _get_obs(self):
    #     data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
    #     return data, self.nonring

class LigninThreeLowTempEval(SetEval):
    def __init__(self):
        super(LigninThreeLowTempEval, self).__init__('lignins_out_high_temp/3_', temp_normal=0.25)

    # def _get_obs(self):
    #     data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
    #     return data, self.nonring

class LigninTwoSet(SetGibbs):
    def __init__(self):
        super(LigninTwoSet, self).__init__('lignins_out/2_', temp_normal=0.5)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class LigninThreeSet(SetGibbs):
    def __init__(self):
        super(LigninThreeSet, self).__init__('lignins_out/3_', temp_normal=0.5)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring


class LigninTwoSetLowTemp(SetGibbs):
    def __init__(self):
        super(LigninTwoSetLowTemp, self).__init__('lignins_out_low_temp/2_', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class LigninThreeSetLowTemp(SetGibbs):
    def __init__(self):
        super(LigninThreeSetLowTemp, self).__init__('lignins_out_low_temp/3_', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class LigninFourSetLowTemp(SetGibbs):
    def __init__(self):
        super(LigninFourSetLowTemp, self).__init__('lignin_four_high_temp/', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class LigninThreeFourHighTempCurriculum(SetCurricula):
    def __init__(self):
        super(LigninThreeFourHighTempCurriculum, self).__init__('lignins_out_high_temp/', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class LigninThreeFourHighTempSet(SetGibbs):
    def __init__(self):
        super(LigninThreeFourHighTempSet, self).__init__('lignins_out_high_temp/', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class LigninFiveSetLowTemp(SetGibbs):
    def __init__(self):
        super(LigninFiveSetLowTemp, self).__init__('lignin_five_high_temp/', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class LigninTwoSet2(SetGibbs):
    def __init__(self):
        super(LigninTwoSet2, self).__init__('lignins_out/2_')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
            return 0.0
        else:
            self.seen.add(tuple(self.action))
            current = confgen.get_conformer_energies(self.mol)[0] * 0.5

            print('standard', self.standard_energy)
            print('current', current)

            if current - self.standard_energy > 10.0:
                return 0.0

            x = current - self.standard_energy
            return (1.0 - x/10.0) / 200.0


class LigninAllSetAdaptive(SetCurricula):
    def __init__(self):
        super(LigninAllSetAdaptive, self).__init__('lignins_out_adaptive_temp/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class DifferentCarbonSkeletonEval(SetEval):
    def __init__(self):
        super(DifferentCarbonSkeletonEval, self).__init__('diff/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

class ThreeSetSkeleton(SetGibbs):
    def __init__(self):
        super(ThreeSetSkeleton, self).__init__('three_set/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

class LigninSevenSetSkeleton(SetGibbs):
    def __init__(self):
        super(LigninSevenSetSkeleton, self).__init__('lignin_obabel_out_low_temp/7_', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class LigninEightSetSkeleton(SetGibbs):
    def __init__(self):
        super(LigninEightSetSkeleton, self).__init__('lignin_obabel_out_low_temp/8_', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class EightLigninEval(SetEval):
    def __init__(self):
        super(EightLigninEval, self).__init__('lignin_obabel_out_low_temp/8_3', temp_normal=0.25)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

class LigninAllSetSkeletonCurriculum(SetCurricula):
    def __init__(self):
        super(LigninAllSetSkeletonCurriculum, self).__init__('lignins_high_temp_curriculum/', temp_normal=0.25, sort_by_size=False)

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton_features(self.mol)])
        return data, self.nonring

