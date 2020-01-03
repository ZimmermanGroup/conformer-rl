import alkanes
from alkanes import *

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

import pdb
import gym
from gym import spaces


confgen = ConformerGeneratorCustom(max_conformers=1,
                                rmsd_threshold=None,
                                force_field="mmff",
                                pool_multiplier=1)


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
    return list(angles)

class Environment(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        pass

    def init(self, data_type, inp):
        #inp is either smiles string or filename
        #datatype can be "file" for .mol file format or "smiles" for smiles format
        if (data_type == "file"):
            m = Chem.MolFromMolFile(inp)
        elif (data_type == "smiles"):
            m = Chem.MolFromSmiles(inp)

        m = Chem.AddHs(m)
        AllChem.EmbedMultipleConfs(m, numConfs=200, numThreads=0)
        energys = confgen.get_conformer_energies(m)

        self.standard_energy = energys.min()
        AllChem.EmbedMultipleConfs(m, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=0)

        self.mol = m
        self.conf = self.mol.GetConformer(id=0)

        self.current_step = 0
        nonring, _ = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]
        self.bonds = self.mol.GetBonds()
        self.angles = getAngles(self.mol)

        self.actionspace = spaces.MultiDiscrete([6 for elt in self.nonring])
        self.observationspace = {}
        self.observationspace['edge'] = gym.Space(shape=[len(self.bonds) + len(self.angles) + len(self.nonring), 10])
        self.observationspace['node'] = gym.Space(shape=[self.mol.GetNumAtoms(), 3])
        

    def _get_reward(self):
        return np.exp(-1.0 * (confgen.get_conformer_energies(self.mol)[0] - self.standard_energy))

    def _get_obs(self):
        obs = {}
        obs['edge'] = np.zeros((len(self.bonds) + len(self.angles) + len(self.nonring), 10))
        obs['node'] = np.array(self.conf.GetPositions())
        
        for bondidx, _ in enumerate(self.bonds):
            bt = self.bonds[bondidx].GetBondType()
            feats = np.array([
                self.bonds[bondidx].GetBeginAtomIdx(),
                self.bonds[bondidx].GetEndAtomIdx(),
                bt == Chem.rdchem.BondType.SINGLE, 
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE, 
                bt == Chem.rdchem.BondType.AROMATIC,
                self.bonds[bondidx].GetIsConjugated(),
                self.bonds[bondidx].IsInRing(),
                0,#angle degree
                0#dihedral degree
            ])
            obs['edge'][bondidx, :] = feats
        for angleidx, _ in enumerate(self.angles):
            feats = np.array([
                self.angles[angleidx][0],
                self.angles[angleidx][2],
                0,
                0,
                0,
                0,
                0,
                0,
                Chem.rdMolTransforms.GetAngleDeg(self.conf, *self.angles[angleidx]),
                0
            ])
            obs['edge'][len(self.bonds) + angleidx, :] = feats
        for dihidx, _ in enumerate(self.nonring):
            feats = np.array([
                self.nonring[dihidx][0],
                self.nonring[dihidx][3],
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                Chem.rdMolTransforms.GetDihedralDeg(self.conf, *self.nonring[dihidx])
            ])
            obs['edge'][len(self.bonds)+len(self.angles)+dihidx, :] = feats
        pdb.set_trace()
        return obs


    def step(self, action):
        #action is shape=[1, len(self.nonring)] array where each element corresponds to the rotation of a dihedral
        print("action is ", action)
        self.action = action
        self.current_step += 1

        desired_torsions = []
        for idx, tors in enumerate(self.nonring):
            ang = -180 + 60 * action[idx]
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

            return obs, rew, done


    def reset(self):
        self.current_step=0
        AllChem.EmbedMultipleConfs(self.mol, numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=0)
        self.conf = self.mol.GetConformer(id=0)
        obs = self._get_obs()

        print('reset called')
        print_torsions(self.mol)
        return obs

env = Environment()
env.init(data_type = "file", inp = 'lignin_guaiacyl.mol')
print(env.reset())

