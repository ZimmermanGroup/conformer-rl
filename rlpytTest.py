import alkanes
from alkanes import *

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import pdb

import gym
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.utils.collections import namedarraytuple
from rlpyt.agents.dqn.epsilon_greedy import EpsilonGreedyAgentMixin
from rlpyt.distributions.epsilon_greedy import EpsilonGreedy



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
    def __init__(self):
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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(250, 3))
        print("Length of action array:", len(self.nonring))
        

    def _get_reward(self):
        return np.exp(-1.0 * (confgen.get_conformer_energies(self.mol)[0] - self.standard_energy))

    def _get_obs(self):
        obs = np.zeros((250, 3))
        obs[0:self.mol.GetNumAtoms(), :] = np.array(self.conf.GetPositions())
        
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

env = GymEnvWrapper(Environment())



#Dummy Neural Net for Testing:
class Net(nn.Module):
    def __init__(self, image_shape, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(750, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, output_size[0])

    def forward(self, x):
        x = x.view(-1, 750)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Env(GymEnvWrapper):

    def __init__(self, e=Environment()):
        super().__init__(env=e)


class Mixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape,
        output_size=env_spaces.action.shape)

AgentInfo = namedarraytuple("AgentInfo", "q")
class CustomAgent(EpsilonGreedyAgentMixin, Mixin, BaseAgent):
    def __init__(self, ModelCls = Net, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B=global_B, env_ranks=env_ranks)
        self.distribution = EpsilonGreedy(dim=env_spaces.action.shape[0])

    def __call__(self, observation, prev_action, prev_reward):
        return self.model(observation)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        action = self.model(observation)
        agent_info = AgentInfo(q=action)
        pdb.set_trace()
        return AgentStep(action=action, agent_info=agent_info)




def build_and_train(run_ID=0, cuda_idx=None):
    sampler = SerialSampler(
        env_kwargs=dict(),
        eval_env_kwargs={},
        EnvCls=Env,
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = DQN(min_steps_learn=1e3)  # Run with defaults.
    agent = CustomAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = {}
    name = "test"
    log_dir = "example_1"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()

build_and_train()