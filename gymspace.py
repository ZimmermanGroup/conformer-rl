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
        self.observationspace['edge'] = gym.Space(shape=[250, 250, 8])
        self.observationspace['node'] = gym.Space(shape=[250, 3])
        

    def _get_reward(self):
        return np.exp(-1.0 * (confgen.get_conformer_energies(self.mol)[0] - self.standard_energy))

    def _get_obs(self):
        obs = {}
        obs['edge'] = np.zeros((250, 250, 8))
        obs['node'] = np.zeros((250, 3))

        obs['node'][0:self.mol.GetNumAtoms(), :] = np.array(self.conf.GetPositions())
        
        for bond in self.bonds:
            bt = bond.GetBondType()
            feats = np.array([
                bt == Chem.rdchem.BondType.SINGLE, 
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE, 
                bt == Chem.rdchem.BondType.AROMATIC,
                bond.GetIsConjugated(),
                bond.IsInRing(),
                0,#angle degree
                0#dihedral degree
            ])
            obs['edge'][bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), :] = feats
        for angle in self.angles:
            feats = np.array([
                0,
                0,
                0,
                0,
                0,
                0,
                Chem.rdMolTransforms.GetAngleDeg(self.conf, *angle),
                0
            ])
            print(Chem.rdMolTransforms.GetAngleDeg(self.conf, *angle) / 180)
            obs['edge'][angle[0], angle[2], :] = feats
        for dih in self.nonring:
            feats = np.array([
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                Chem.rdMolTransforms.GetDihedralDeg(self.conf, *dih)
            ])
            print(Chem.rdMolTransforms.GetDihedralDeg(self.conf, *dih) / 180)
            obs['edge'][dih[0], dih[3], :] = feats
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

