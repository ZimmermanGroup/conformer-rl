import alkanes
from alkanes import *

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing
import logging
import torch
import pandas as pd
import time
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance


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


confgen = ConformerGeneratorCustom(max_conformers=1,
                             rmsd_threshold=None,
                             force_field='mmff',
                             pool_multiplier=1)

m = Chem.MolFromMolFile('lignin_guaiacyl.mol')
m = Chem.AddHs(m)
AllChem.EmbedMultipleConfs(m, numConfs=200, numThreads=0)
res = AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=0)

energys = confgen.get_conformer_energies(m)


class LigninEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(LigninEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
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


mm = Chem.MolFromSmiles('CCC(CC)CC(CCC)CCC')
mm = Chem.AddHs(mm)
AllChem.EmbedMultipleConfs(mm, numConfs=200, numThreads=0)
res = AllChem.MMFFOptimizeMoleculeConfs(mm, numThreads=0)

benergys = confgen.get_conformer_energies(mm)


class BranchedCarbon(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BranchedCarbon, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.standard_energy = benergys.min()
        AllChem.EmbedMultipleConfs(mm, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(mm, numThreads=0)

        self.mol = mm
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

mmm = Chem.MolFromMolFile('258-trihexyl-decane.mol')
mmm = Chem.AddHs(mmm)
AllChem.EmbedMultipleConfs(mmm, numConfs=200, numThreads=0)
res = AllChem.MMFFOptimizeMoleculeConfs(mmm, numThreads=0)

tenergys = confgen.get_conformer_energies(mmm)


class Trihexyl(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Trihexyl, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.standard_energy = tenergys.min()
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


mmmm = Chem.MolFromSmiles('CC(CCC)CCCC(CCCC)CC')
mmmm = Chem.AddHs(mmmm)
AllChem.EmbedMultipleConfs(mmmm, numConfs=200, numThreads=0)
res = AllChem.MMFFOptimizeMoleculeConfs(mmmm, numThreads=0)

benergys = confgen.get_conformer_energies(mmmm)


class DifferentCarbon(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DifferentCarbon, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.standard_energy = benergys.min()
        AllChem.EmbedMultipleConfs(mmmm, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(mmmm, numThreads=0)

        self.mol = mmmm
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


class DifferentCarbonSet(DifferentCarbon):
    def __init__(self):
        super(DifferentCarbonSet, self).__init__()
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
        return super(DifferentCarbonSet, self).reset()
