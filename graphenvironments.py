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

confgen = ConformerGeneratorCustom(max_conformers=1,
                             rmsd_threshold=None,
                             force_field='mmff',
                             pool_multiplier=1)

class SetGibbs(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, folder_name, gibbs_normalize=True, in_order=False):
        super(SetGibbs, self).__init__()
        self.gibbs_normalize = gibbs_normalize
        self.all_files = glob.glob(f'{folder_name}*')
        self.all_files.sort(key=os.path.getsize)
        self.in_order = in_order
        self.choice = -1
        self.episode_reward = 0
        self.choice_ind = 1
        self.num_good_episodes = 0

        while True:
            obj = self.molecule_choice()
            self.mol = Chem.MolFromSmiles(obj['mol'])

            self.standard_energy = float(obj['standard'])
            if 'total' in obj and self.gibbs_normalize:
                self.total = obj['total']
            else:
                self.total = 1.0
            self.mol = Chem.AddHs(self.mol)
            res = AllChem.EmbedMultipleConfs(self.mol)
            if not len(res):
                continue
            res = Chem.AllChem.MMFFOptimizeMoleculeConfs(self.mol)
            self.conf = self.mol.GetConformer(id=0)
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

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
            self.repeats += 1
            return 0
        else:
            self.seen.add(tuple(self.action))
            current = confgen.get_conformer_energies(self.mol)[0]
            print('standard', self.standard_energy)
            print('current', current)
            return np.exp(-1.0 * (current - self.standard_energy)) / self.total

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecsimple(self.mol)])
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

        obs = self._get_obs()
        rew = self._get_reward()
        self.episode_reward += rew

        rbn = len(self.nonring)
        if rbn == 3:
            done = (self.current_step == 25)
        else:
            done = (self.current_step == 200)

        print(done)
        self.mol_appends(done)

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
            self.mol = Chem.MolFromSmiles(obj['mol'])
            self.standard_energy = float(obj['standard'])
            if 'total' in obj and self.gibbs_normalize:
                self.total = obj['total']
            else:
                self.total = 1.0
            self.mol = Chem.AddHs(self.mol)
            res = AllChem.EmbedMultipleConfs(self.mol, numConfs=1)
            if not len(res):
                continue
            res = Chem.AllChem.MMFFOptimizeMoleculeConfs(self.mol)
            self.conf = self.mol.GetConformer(id=0)
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


class SetEval(SetGibbs):
    def mol_appends(self, done):
        if done:
            import pickle
            with open('test_mol.pickle', 'wb') as fp:
                pickle.dump(self.mol, fp)
        else:
            c = self.mol.GetConformer(id=0)
            self.mol.AddConformer(c, assignId=True)


class TrihexylEval(SetEval):
    def __init__(self):
        super(TrihexylEval, self).__init__('trihexyl/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecstupidsimple(self.mol)])
        return data, self.nonring


class SetCurricula(SetGibbs):
    def info(self, info):
        info['num_good_episodes'] = self.num_good_episodes
        info['choice_ind'] = self.choice_ind
        return info

    def molecule_choice(self):
        if self.episode_reward > 0.85:
            self.num_good_episodes += 1
        else:
            self.num_good_episodes = 0

        if self.num_good_episodes >= 10:
            self.choice_ind *= 2
            self.choice_ind = self.choice_ind % len(self.all_files)
            self.num_good_episodes = 0

        if self.in_order:
            self.choice = (self.choice + 1) % len(self.all_files[0:self.choice_ind])
            cjson = self.all_files[0:self.choice_ind][self.choice]
        else:
            cjson = np.random.choice(self.all_files[0:self.choice_ind])

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

class TenTorsionSetCurriculumSimple(SetCurricula):
    def __init__(self):
        super(TenTorsionSetCurriculumSimple, self).__init__('huge_hc_set/10_')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecstupidsimple(self.mol)])
        return data, self.nonring

class TenTorsionSetCurriculumBasic(SetCurricula):
    def __init__(self):
        super(TenTorsionSetCurriculumBasic, self).__init__('huge_hc_set/10_')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecbasic(self.mol)])
        return data, self.nonring

class TenTorsionSetCurriculum(SetCurricula):
    def __init__(self):
        super(TenTorsionSetCurriculum, self).__init__('huge_hc_set/10_')

class DifferentCarbonSet(SetGibbs):
    def __init__(self):
        super(DifferentCarbonSet, self).__init__('diff/')

class DifferentCarbonSetStupid(SetGibbsStupid):
    def __init__(self):
        super(DifferentCarbonSetStupid, self).__init__('diff/')


class Trihexyl(SetGibbs):
    def __init__(self):
        super(Trihexyl, self).__init__('trihexyl/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecbasic(self.mol)])
        return data, self.nonring

class LargeCarbonSet(SetGibbsStupid):
    def __init__(self):
        super(LargeCarbonSet, self).__init__('large_carbon/')

class DifferentCarbonSetDense(SetGibbsDense):
    def __init__(self):
        super(DifferentCarbonSetDense, self).__init__('diff/')

class TestSet(SetGibbs):
    def __init__(self):
        super(TestSet, self).__init__('test_set', gibbs_normalize=False)

class InOrderTestSet(SetGibbs):
    def __init__(self):
        super(InOrderTestSet, self).__init__('test_set', in_order=True)

class SetEnergy(SetGibbs):
    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
            return 0.0
        else:
            self.seen.add(tuple(self.action))
            print('standard', self.standard_energy)
            current = confgen.get_conformer_energies(self.mol)[0]
            print('current', current)
            if current - self.standard_energy > 50.0:
                return 0.0
            return self.standard_energy / (200 * current)

class SetEnergyScaled(SetGibbs):
    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
            return -1.0
        else:
            self.seen.add(tuple(self.action))
            current = confgen.get_conformer_energies(self.mol)[0]
            print('standard', self.standard_energy)
            print('current', current)
            if current - self.standard_energy > 10.0:
                return -1.0

            x = current - self.standard_energy
            return 1.0 - x/5.0

class GiantSet(SetGibbs):
    def __init__(self):
        super(GiantSet, self).__init__('giant_hc_set/')

class OneSet(SetGibbs):
    def __init__(self):
        super(OneSet, self).__init__('one_set/')

class TwoSet(SetGibbs):
    def __init__(self):
        super(TwoSet, self).__init__('two_set/')

class ThreeSet(SetGibbs):
    def __init__(self):
        super(ThreeSet, self).__init__('three_set/')

class ThreeSetSimple(SetGibbsStupid):
    def __init__(self):
        super(ThreeSetSimple, self).__init__('three_set/')

class FourSet(SetGibbs):
    def __init__(self):
        super(FourSet, self).__init__('four_set/')

class LigninSet(SetGibbsStupid):
    def __init__(self):
        super(LigninSet, self).__init__('lignin_large/')

class LigninTest(SetGibbsStupid):
    def __init__(self):
        super(LigninTest, self).__init__('lignin_ten/')

class LigninSmalls(SetCurricula):
    def __init__(self):
        super(LigninSmalls, self).__init__('LigninSmallSet/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecbasic(self.mol)])
        return data, self.nonring

class LigninFourSet(SetGibbs):
    def __init__(self):
        super(LigninFourSet, self).__init__('LigninFourSet/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecbasic(self.mol)])
        return data, self.nonring

class LigninThreeSet(SetGibbs):
    def __init__(self):
        super(LigninThreeSet, self).__init__('three_lignin/')


    def _get_obs(self):
        data = Batch.from_data_list([mol2vecbasic(self.mol)])
        return data, self.nonring

class LigninTwoSet(SetGibbs):
    def __init__(self):
        super(LigninTwoSet, self).__init__('lignin_two/')

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecskeleton(self.mol)])
        return data, self.nonring

# def _get_reward(self):
#     print('standard_energy', self.standard_energy)
#     return self.standard_energy / confgen.get_conformer_energies(self.mol)[0]
