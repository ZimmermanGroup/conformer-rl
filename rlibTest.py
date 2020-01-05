import alkanes
from alkanes import *

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

import pdb

import gym
from gym import spaces

import ray
import ray.rllib.agents
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import ray.rllib.models.catalog as catalog

from ray.rllib.utils import try_import_tf
tf = try_import_tf()

from ray import tune


confgen = ConformerGeneratorCustom(max_conformers=1,
                                rmsd_threshold=None,
                                force_field="mmff",
                                pool_multiplier=1)

m = Chem.MolFromMolFile('lignin_guaiacyl.mol')


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
    def __init__(self, env_config):
        mol = Chem.AddHs(m)
        AllChem.EmbedMultipleConfs(mol, numConfs=200, numThreads=0)
        energys = confgen.get_conformer_energies(mol)

        self.standard_energy = energys.min()
        AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=0)
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)

        self.mol = mol
        self.conf = self.mol.GetConformer(id=0)

        self.current_step = 0
        nonring, _ = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]
        self.bonds = self.mol.GetBonds()
        self.angles = getAngles(self.mol)

        self.action_space = spaces.MultiDiscrete([6 for elt in self.nonring])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 250, 3))
        print("Length of action array:", len(self.nonring))
        

    def _get_reward(self):
        return np.exp(-1.0 * (confgen.get_conformer_energies(self.mol)[0] - self.standard_energy))

    def _get_obs(self):
        obs = np.zeros((250, 3))
        obs[0:self.mol.GetNumAtoms(), :] = np.array(self.conf.GetPositions())
        obs = np.reshape(obs, (1, 250, 3))
        
        return obs


    def step(self, action):
        #action is shape=[1, len(self.nonring)] array where each element corresponds to the rotation of a dihedral
        print("action is ", action)
        self.action = action
        self.current_step += 1

        desired_torsions = []
        for idx, tors in enumerate(self.nonring):
            ang = -180 + 60 * action[idx]
            ang = ang.item()
            desired_torsions.append(ang)
            Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], ang)

            ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol))
            ff.Initialize()
            ff.Minimize()


            obs = self._get_obs()
            rew = self._get_reward()
            done = self.current_step == 100

            print("step is: ", self.current_step)
            print("reward is ", rew)
            print ("new state is:")
            print_torsions(self.mol)

            return obs, rew, done, {}


    def reset(self):
        self.current_step=0
        AllChem.EmbedMultipleConfs(self.mol, numConfs=1, numThreads=0)
        AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=0)
        self.conf = self.mol.GetConformer(id=0)
        obs = self._get_obs()

        print('reset called')
        print_torsions(self.mol)
        return obs


class Fcn(TFModelV2):
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(Fcn, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.model = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name)

        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()





ray.init()
ModelCatalog.register_custom_model("fullyconnected", Fcn)

#standard implementation:
"""
modelconfig = catalog.MODEL_DEFAULTS.copy()
modelconfig["custom_model"] = "fullyconnected"
#modelconfig["fcnet_hiddens"] = [24, 1]
config = ppo.DEFAULT_CONFIG.copy()
config["model"] = modelconfig

trainer = ppo.PPOTrainer(config = config, env=Environment)
for i in range(1):
    result = trainer.train()
    print(pretty_print(result))
"""
#tune implementation:
tune.run(
    "PPO",
    stop={"time_total_s": 2},
    config={
        "env": Environment,
        "model": {
            "custom_model": "fullyconnected",
        }
    }
)