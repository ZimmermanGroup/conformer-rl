from utils import *

import numpy as np
import scipy

from rdkit import Chem
from rdkit.Chem import AllChem

import multiprocessing
import logging
import torch
import pandas as pd
import time

import torch

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance

import glob
import json

import alkanes
from alkanes import *

import pdb

def bond_features(bond, use_chirality=False):
    from rdkit import Chem
    bt = bond.GetBondType()
    bond_feats = [
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


def bond_features_meta(bond, torsionAngle = 0, bondAngle = 0, use_chirality=False):
    from rdkit import Chem
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
        torsionAngle,
        bondAngle
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)


def getAngles(mol): #returns a list of all sets of three atoms involved in an angle (no repeated angles).
    angles = set()
    bondDict = {}
    bonds = mol.GetBonds()
    for bond in bonds:
        if not bond.IsInRing():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if start in bondDict:
                for atom in bondDict[start]:
                    if atom != start and atom != end:
                        if (atom < end):
                            angles.add((atom, start, end))
                        elif end < atom:
                            angles.add((end, start, atom))
                bondDict[start].append(end)
            else:
                bondDict[start] = [end]
            if end in bondDict:
                for atom in bondDict[end]:
                    if atom != start and atom != end:
                        if atom < start:
                            angles.add((atom, end, start))
                        elif start < atom:
                            angles.add((start, end, atom))
                bondDict[end].append(start)
    return angles

#################
# pen added
#################
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

def mol2points(mol):
    conf = mol.GetConformer(id=-1)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [atom_features_simple(atom, conf) for atom in atoms]

    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
            )
    data = Distance()(data)
    return data


def mol2vecmeta(mol): #includes all the meta-edge features, eg. angles and dihedrals
    conf = mol.GetConformer(id=-1)
    nonring, _ = TorsionFingerprints.CalculateTorsionLists(mol)
    angles = getAngles(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [atom_features_simple(atom, conf) for atom in atoms]
    #edge_index = get_bond_pair(mol)

    #Index-consistent way of adding things into edge_index to match the indexes of edge_attr:
    edge_index = [[],[]]
    for bond in bonds:
        #edge_index[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        #edge_index[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
        edge_index[0].append(bond.GetBeginAtomIdx())
        edge_index[1].append(bond.GetEndAtomIdx())

    edge_attr = [bond_features_meta(bond, use_chirality=False) for bond in bonds]
    for torsion in nonring:
        #edge_index[0] += [torsion[0][0][0], torsion[0][0][3]]
        #edge_index[1] += [torsion[0][0][3], torsion[0][0][0]]
        edge_index[0].append(torsion[0][0][0])
        edge_index[1].append(torsion[0][0][3])
        edge_attr.append([
            0, 0, 0, 0, 0, 0, torsion[1], -1
        ])
    for angle in angles:
        #edge_index[0] += [angle[0], angle[2]]
        #edge_index[1] += [angle[2], angle[0]]
        edge_index[0].append(angle[0])
        edge_index[1].append(angle[2])
        edge_attr.append([
            0, 0, 0, 0, 0, 0, -1, Chem.rdMolTransforms.GetAngleDeg(conf, *angle)
        ])
    for bond in bonds:
        edge_index[1].append(bond.GetBeginAtomIdx())
        edge_index[0].append(bond.GetEndAtomIdx())
        edge_attr.append(bond_features_meta(bond))
    for torsion in nonring:
        edge_index[1].append(torsion[0][0][0])
        edge_index[0].append(torsion[0][0][3])
        edge_attr.append([
            0, 0, 0, 0, 0, 0, -1*torsion[1], -1
        ])
    for angle in angles:
        edge_index[1].append(angle[0])
        edge_index[0].append(angle[2])
        edge_attr.append([
            0, 0, 0, 0, 0, 0, -1, Chem.rdMolTransforms.GetAngleDeg(conf, *angle)
        ])
    data = Data(
                x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                pos=torch.Tensor(conf.GetPositions())
            )
    data = Distance()(data)
    return data


confgen = ConformerGeneratorCustom(max_conformers=1,
                             rmsd_threshold=None,
                             force_field='mmff',
                             pool_multiplier=1)


lignin_standard = 0
res = None#glob.glob('cache/lignin.json')

if res:
    with open(res[0]) as fp:
        cached = json.load(fp)
        lignin_standard = cached['standard_energy']
else:
    m = Chem.MolFromMolFile('lignin_guaiacyl.mol')
    m = Chem.AddHs(m)
    AllChem.EmbedMultipleConfs(m, numConfs=200, numThreads=-1)
    res = AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=-1)

    energys = confgen.get_conformer_energies(m)
    cached = {}
    lignin_standard = energys.min()
    cached['standard_energy'] = lignin_standard
    with open('cache/lignin.json', 'w') as fp:
        json.dump(cached, fp)


class LigninEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(LigninEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when
        # using discrete actions:
        self.standard_energy = energys.min()
        AllChem.EmbedMultipleConfs(m, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=0)

        self.mol = m
        self.conf = self.mol.GetConformer(id=0)
        self.everseen = set()
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]
        self.delta_t = []

    def _get_reward(self):
        return np.exp(-1.0 * (confgen.get_conformer_energies(self.mol)[0] - self.standard_energy))

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecsimple(self.mol)])
        #data = Batch.from_data_list([mol2vecmeta(self.mol)])
        return data, self.nonring

    def step(self, action):
        # Execute one time step within the environment
        print("action is ", action)
        self.action = action
        self.current_step += 1

        begin_step = time.process_time()
        desired_torsions = []

#         ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(m, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(m))
#         for idx, tors in enumerate(self.nonring):
#             deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
#             ang = -180.0 + 60 * action[idx]
#             desired_torsions.append(ang)
#             ff.MMFFAddTorsionConstraint(*tup, False, ang, ang,  1e12)

#         ff.Initialize()
#         ff.Minimize()


        for idx, tors in enumerate(self.nonring):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * action[idx]
            desired_torsions.append(ang)
            Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], ang)

        # degs = [Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors) for tors in self.nonring]
        #
        # dist = np.linalg.norm(np.sin(np.array(degs) * np.pi / 180.) - np.sin(np.array(desired_torsions) * np.pi / 180.))
        # dist += np.linalg.norm(np.cos(np.array(degs)* np.pi / 180.) - np.cos(np.array(desired_torsions) * np.pi / 180.))
        #
        # if dist > 0.1:
        #     print('desired torsions', desired_torsions)
        #     print('actual torsions', degs)
        #     print(dist)
        #     energy = np.exp(-1.0 * confgen.get_conformer_energies(m)[0])
        #     print(energy)
        #
        #     raise Exception

        ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol))
        ff.Initialize()
        ff.Minimize()


        obs = self._get_obs()
        rew = self._get_reward()
        done = self.current_step == 200


        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)


        end_step = time.process_time()

        delta_t = end_step-begin_step
        self.delta_t.append(delta_t)

        return obs, rew, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        AllChem.EmbedMultipleConfs(self.mol, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=0)
        obs = self._get_obs()
        self.conf = self.mol.GetConformer(id=0)

        print('step time mean', np.array(self.delta_t).mean())
        print('reset called')
        print_torsions(self.mol)
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print_torsions(self.mol)



class LigninSetEnv(LigninEnv):
    def __init__(self):
        super(LigninSetEnv, self).__init__()
        self.seen = set()

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
            return 0
        else:
            self.seen.add(tuple(self.action))
            return np.exp(-1.0 * (confgen.get_conformer_energies(self.mol)[0] - self.standard_energy))

    def reset(self):
        self.seen = set()
        return super(LigninSetEnv, self).reset()

test = LigninEnv()
print(test.reset())


mm = Chem.MolFromSmiles('CCC(CC)CC(CCC)CCC')
mm = Chem.AddHs(mm)
AllChem.EmbedMultipleConfs(mm, numConfs=200, numThreads=0)
res = AllChem.MMFFOptimizeMoleculeConfs(mm, numThreads=0)

AllChem.EmbedMultipleConfs(mm, numConfs=200, numThreads=-1)
res = AllChem.MMFFOptimizeMoleculeConfs(mm, numThreads=-1)

energys = confgen.get_conformer_energies(mm)

cached = {}
branched_standard = energys.min()
cached['standard_energy'] = branched_standard
with open('cache/branched.json', 'w') as fp:
    json.dump(cached, fp)

class BranchedCarbon(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BranchedCarbon, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.standard_energy = branched_standard
        self.mol = Chem.MolFromSmiles('CCC(CC)CC(CCC)CCC')
        self.mol = Chem.AddHs(self.mol)

        AllChem.EmbedMultipleConfs(self.mol, numConfs=1, numThreads=-1)
        res = AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=-1)

        self.conf = self.mol.GetConformer(id=0)
        self.everseen = set()
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]
        self.delta_t = []

    def _get_reward(self):
        print('standard_energy', self.standard_energy)
        return np.exp(-1.0 * (confgen.get_conformer_energies(self.mol)[0] - self.standard_energy))

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecsimple(self.mol)])
        return data, self.nonring

    def step(self, action):
        # Execute one time step within the environment
        print("action is ", action)
        self.action = action
        self.current_step += 1

        begin_step = time.process_time()
        desired_torsions = []

#         ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(m, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(m))
#         for idx, tors in enumerate(self.nonring):
#             deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
#             ang = -180.0 + 60 * action[idx]
#             desired_torsions.append(ang)
#             ff.MMFFAddTorsionConstraint(*tup, False, ang, ang,  1e12)

#         ff.Initialize()
#         ff.Minimize()


        for idx, tors in enumerate(self.nonring):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * action[idx]
            desired_torsions.append(ang)
            Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], ang)



        # degs = [Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors) for tors in self.nonring]
        #
        # dist = np.linalg.norm(np.sin(np.array(degs) * np.pi / 180.) - np.sin(np.array(desired_torsions) * np.pi / 180.))
        # dist += np.linalg.norm(np.cos(np.array(degs)* np.pi / 180.) - np.cos(np.array(desired_torsions) * np.pi / 180.))
        #
        # if dist > 0.1:
        #     print('desired torsions', desired_torsions)
        #     print('actual torsions', degs)
        #     print(dist)
        #     energy = np.exp(-1.0 * confgen.get_conformer_energies(m)[0])
        #     print(energy)
        #
        #     raise Exception

        ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol))
        ff.Initialize()
        ff.Minimize()


        obs = self._get_obs()
        rew = self._get_reward()
        done = self.current_step == 200


        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)


        end_step = time.process_time()

        delta_t = end_step-begin_step
        self.delta_t.append(delta_t)

        return obs, rew, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        AllChem.EmbedMultipleConfs(self.mol, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=0)
        obs = self._get_obs()
        self.conf = self.mol.GetConformer(id=0)

        print('step time mean', np.array(self.delta_t).mean())
        print('reset called')
        print_torsions(self.mol)
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print_torsions(self.mol)


class BranchedCarbonSet(BranchedCarbon):
    def __init__(self):
        super(BranchedCarbonSet, self).__init__()
        self.seen = set()

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
            return 0
        else:
            self.seen.add(tuple(self.action))
            print('standard_energy', self.standard_energy)
            return np.exp(-1.0 * (confgen.get_conformer_energies(self.mol)[0] - self.standard_energy))

    def reset(self):
        self.seen = set()
        return super(BranchedCarbonSet, self).reset()


trihexyl_standard = 0
mmm = Chem.MolFromMolFile('258-trihexyl-decane.mol')
mmm = Chem.AddHs(mmm)

res = glob.glob('cache/trihexyl.json')

if res:
    with open(res[0]) as fp:
        cached = json.load(fp)
        trihexyl_standard = cached['standard_energy']
else:
    AllChem.EmbedMultipleConfs(mmm, numConfs=200, numThreads=0)
    res = AllChem.MMFFOptimizeMoleculeConfs(mmm, numThreads=0)

    tenergys = confgen.get_conformer_energies(mmm)
    cached = {}
    trihexyl_standard = tenergys.min()
    cached['standard_energy'] = trihexyl_standard
    with open('cache/trihexyl.json', 'w') as fp:
        json.dump(cached, fp)

class Trihexyl(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Trihexyl, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.standard_energy = trihexyl_standard
        AllChem.EmbedMultipleConfs(mmm, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(mmm, numThreads=0)

        self.mol = mmm
        self.conf = self.mol.GetConformer(id=0)
        self.everseen = set()
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]
        self.delta_t = []

    def _get_reward(self):
        return np.exp(-1.0 * (confgen.get_conformer_energies(self.mol)[0] - self.standard_energy))

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecsimple(self.mol)])
        return data, self.nonring

    def step(self, action):
        # Execute one time step within the environment
        print("action is ", action)
        self.action = action
        self.current_step += 1

        begin_step = time.process_time()
        desired_torsions = []

#         ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(m, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(m))
#         for idx, tors in enumerate(self.nonring):
#             deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
#             ang = -180.0 + 60 * action[idx]
#             desired_torsions.append(ang)
#             ff.MMFFAddTorsionConstraint(*tup, False, ang, ang,  1e12)

#         ff.Initialize()
#         ff.Minimize()


        for idx, tors in enumerate(self.nonring):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * action[idx]
            desired_torsions.append(ang)
            Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], ang)



        # degs = [Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors) for tors in self.nonring]
        #
        # dist = np.linalg.norm(np.sin(np.array(degs) * np.pi / 180.) - np.sin(np.array(desired_torsions) * np.pi / 180.))
        # dist += np.linalg.norm(np.cos(np.array(degs)* np.pi / 180.) - np.cos(np.array(desired_torsions) * np.pi / 180.))
        #
        # if dist > 0.1:
        #     print('desired torsions', desired_torsions)
        #     print('actual torsions', degs)
        #     print(dist)
        #     energy = np.exp(-1.0 * confgen.get_conformer_energies(m)[0])
        #     print(energy)
        #
        #     raise Exception

        ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol))
        ff.Initialize()
        ff.Minimize()


        obs = self._get_obs()
        rew = self._get_reward()
        done = self.current_step == 200


        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)


        end_step = time.process_time()

        delta_t = end_step-begin_step
        self.delta_t.append(delta_t)

        return obs, rew, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        AllChem.EmbedMultipleConfs(self.mol, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=0)
        obs = self._get_obs()
        self.conf = self.mol.GetConformer(id=0)

        print('step time mean', np.array(self.delta_t).mean())
        print('reset called')
        print_torsions(self.mol)
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print_torsions(self.mol)


class TrihexylSet(Trihexyl):
    def __init__(self):
        super(TrihexylSet, self).__init__()
        self.seen = set()

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
            return 0
        else:
            self.seen.add(tuple(self.action))
            return np.exp(-1.0 * (confgen.get_conformer_energies(self.mol)[0] - self.standard_energy))

    def reset(self):
        self.seen = set()
        return super(TrihexylSet, self).reset()

res = glob.glob('cache/differentcarbon.json')
diffcarbon_standard = 0
if res:
    with open(res[0]) as fp:
        cached = json.load(fp)
        diffcarbon_standard = cached['standard_energy']
else:
    mmmm = Chem.MolFromSmiles('CC(CCC)CCCC(CCCC)CC')
    mmmm = Chem.AddHs(mmmm)
    AllChem.EmbedMultipleConfs(mmmm, numConfs=200, numThreads=0)
    res = AllChem.MMFFOptimizeMoleculeConfs(mmmm, numThreads=0)
    benergys = confgen.get_conformer_energies(mmmm)

    diffcarbon_standard = benergys.min()
    cached = {}
    cached['standard_energy'] = diffcarbon_standard
    with open('cache/differentcarbon.json', 'w') as fp:
        json.dump(cached, fp)

class DifferentCarbon(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DifferentCarbon, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.standard_energy = diffcarbon_standard
        self.mol = Chem.MolFromSmiles('CC(CCC)CCCC(CCCC)CC')
        self.mol = Chem.AddHs(self.mol)
        AllChem.EmbedMultipleConfs(self.mol, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=0)

        self.conf = self.mol.GetConformer(id=0)
        self.everseen = set()
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]
        self.delta_t = []

    def _get_reward(self):
        print('standard_energy', self.standard_energy)
        return np.exp(-1.0 * (confgen.get_conformer_energies(self.mol)[0] - self.standard_energy))

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecsimple(self.mol)])
        return data, self.nonring

    def step(self, action):
        # Execute one time step within the environment
        print("action is ", action)
        self.action = action
        self.current_step += 1

        begin_step = time.process_time()
        desired_torsions = []

        for idx, tors in enumerate(self.nonring):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * action[idx]
            desired_torsions.append(ang)
            Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], ang)

        ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol))
        ff.Initialize()
        ff.Minimize()

        obs = self._get_obs()
        rew = self._get_reward()
        done = self.current_step == 200

        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)

        end_step = time.process_time()

        delta_t = end_step-begin_step
        self.delta_t.append(delta_t)

        return obs, rew, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        AllChem.EmbedMultipleConfs(self.mol, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=0)
        obs = self._get_obs()
        self.conf = self.mol.GetConformer(id=0)

        print('step time mean', np.array(self.delta_t).mean())
        print('reset called')
        print_torsions(self.mol)
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print_torsions(self.mol)


class DifferentCarbonSet(DifferentCarbon):
    def __init__(self):
        super(DifferentCarbonSet, self).__init__()
        self.seen = set()
        self.m = Chem.MolFromSmiles('CC(CCC)CCCC(CCCC)CC')
        self.m = Chem.AddHs(self.m)

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
            return 0.0
        else:
            self.seen.add(tuple(self.action))
            return super(DifferentCarbonSet, self)._get_reward()

    def reset(self):
        self.seen = set()
        return super(DifferentCarbonSet, self).reset()


class SetGibbs(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, folder_name):
        super(SetGibbs, self).__init__()

        self.all_files = glob.glob(f'{folder_name}/*.json')

        while True:
            cjson = np.random.choice(self.all_files[:-1])
            with open(cjson) as fp:
                obj = json.load(fp)

            self.mol = Chem.MolFromSmiles(obj['mol'])

            self.standard_energy = float(obj['standard'])
            if 'total' in obj:
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

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
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
        self.action = action
        self.current_step += 1

        begin_step = time.process_time()
        desired_torsions = []

        for idx, tors in enumerate(self.nonring):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * action[idx]
            desired_torsions.append(ang)
            try:
                Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], float(ang))
            except:
                Chem.MolToMolFile(self.mol, 'debug.mol')
                exit()

        Chem.AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=-1)

        obs = self._get_obs()
        rew = self._get_reward()
        done = self.current_step == 200

        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)

        end_step = time.process_time()

        delta_t = end_step-begin_step
        self.delta_t.append(delta_t)

        return obs, rew, done, {}

    def reset(self):
        self.current_step = 0

        while True:
            cjson = np.random.choice(self.all_files[:-1])
            with open(cjson) as fp:
                obj = json.load(fp)

            self.mol = Chem.MolFromSmiles(obj['mol'])
            self.standard_energy = float(obj['standard'])
            if 'total' in obj:
                self.total = obj['total']
            else:
                self.total = 1.0
            self.mol = Chem.AddHs(self.mol)
            res = AllChem.EmbedMultipleConfs(self.mol, numConfs=1)
            if not len(res):
                continue
            res = Chem.AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=-1)
            self.conf = self.mol.GetConformer(id=0)
            break

        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]

        obs = self._get_obs()

        print('step time mean', np.array(self.delta_t).mean())
        print('reset called')
        print_torsions(self.mol)
        return obs

    def render(self, mode='human', close=False):
        print_torsions(self.mol)

class UnholySet(SetGibbs):
    def __init__(self):
        super(UnholySet, self).__init__('branched_hydrocarbons')

class UnholierSet(SetGibbs):
    def __init__(self):
        super(UnholierSet, self).__init__('bigger_branched')

class TestSet(SetGibbs):
    def __init__(self):
        super(TestSet, self).__init__('test_set')

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
            if current - self.standard_energy > 10.0:
                return 0.0
            return self.standard_energy / current

class SetEnergyEval(SetEnergy):
    def __init__(self, folder_name):
        super(SetEnergyEval, self).__init__(folder_name, eval=True)

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
            return 0.0
        else:
            self.seen.add(tuple(self.action))
            print('standard', self.standard_energy)
            current = confgen.get_conformer_energies(self.mol)[0]
            print('current', current)
            return np.exp(-1.0 * (current - self.standard_energy))

class UnholierSetEnergyEval(SetEnergyEval):
    def __init__(self):
        super(UnholierSetEnergyEval, self).__init__('bigger_branched')

class UnholySetEnergy(SetEnergy):
    def __init__(self):
        super(UnholySetEnergy, self).__init__('branched_hydrocarbons')

class UnholierSetEnergy(SetEnergy):
    def __init__(self):
        super(UnholierSetEnergy, self).__init__('bigger_branched')

class UnholierSetEnergyScaled(UnholierSetEnergy):
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

class UnholierPointSetEnergy(UnholierSetEnergy):
    def _get_obs(self):
        data = Batch.from_data_list([mol2points(self.mol)])
        return data, self.nonring

class DifferentCarbonReward(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DifferentCarbonReward, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.standard_energy = diffcarbon_standard
        self.mol = Chem.MolFromSmiles('CC(CCC)CCCC(CCCC)CC')
        self.mol = Chem.AddHs(self.mol)
        AllChem.EmbedMultipleConfs(self.mol, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=0)

        self.conf = self.mol.GetConformer(id=0)
        self.everseen = set()
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]
        self.delta_t = []

    def _get_reward(self):
        print('standard_energy', self.standard_energy)
        return self.standard_energy / confgen.get_conformer_energies(self.mol)[0]

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecsimple(self.mol)])
        return data, self.nonring

    def step(self, action):
        # Execute one time step within the environment
        print("action is ", action)
        self.action = action
        self.current_step += 1

        begin_step = time.process_time()
        desired_torsions = []

#         ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(m, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(m))
#         for idx, tors in enumerate(self.nonring):
#             deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
#             ang = -180.0 + 60 * action[idx]
#             desired_torsions.append(ang)
#             ff.MMFFAddTorsionConstraint(*tup, False, ang, ang,  1e12)

#         ff.Initialize()
#         ff.Minimize()


        for idx, tors in enumerate(self.nonring):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * action[idx]
            desired_torsions.append(ang)
            Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], ang)



        # degs = [Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors) for tors in self.nonring]
        #
        # dist = np.linalg.norm(np.sin(np.array(degs) * np.pi / 180.) - np.sin(np.array(desired_torsions) * np.pi / 180.))
        # dist += np.linalg.norm(np.cos(np.array(degs)* np.pi / 180.) - np.cos(np.array(desired_torsions) * np.pi / 180.))
        #
        # if dist > 0.1:
        #     print('desired torsions', desired_torsions)
        #     print('actual torsions', degs)
        #     print(dist)
        #     energy = np.exp(-1.0 * confgen.get_conformer_energies(m)[0])
        #     print(energy)
        #
        #     raise Exception

        ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol))
        ff.Initialize()
        ff.Minimize()


        obs = self._get_obs()
        rew = self._get_reward()
        done = self.current_step == 200


        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)


        end_step = time.process_time()

        delta_t = end_step-begin_step
        self.delta_t.append(delta_t)

        return obs, rew, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        AllChem.EmbedMultipleConfs(self.mol, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=0)
        obs = self._get_obs()
        self.conf = self.mol.GetConformer(id=0)

        print('step time mean', np.array(self.delta_t).mean())
        print('reset called')
        print_torsions(self.mol)
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print_torsions(self.mol)


class DifferentCarbonSetReward(DifferentCarbonReward):
    def __init__(self):
        super(DifferentCarbonSetReward, self).__init__()
        self.seen = set()
        self.m = Chem.MolFromSmiles('CC(CCC)CCCC(CCCC)CC')
        self.m = Chem.AddHs(self.m)

    def _get_reward(self):
        if tuple(self.action) in self.seen:
            print('already seen')
            return 0.0
        else:
            self.seen.add(tuple(self.action))
            return super(DifferentCarbonSetReward, self)._get_reward()

    def reset(self):
        self.seen = set()
        return super(DifferentCarbonSet, self).reset()

