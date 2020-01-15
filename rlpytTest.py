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
from torch.autograd import Variable

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.algos.pg.ppo import PPO
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.utils.collections import namedarraytuple
from rlpyt.distributions.categorical import DistInfo
from rlpyt.utils.buffer import buffer_to

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance
import torch_geometric.nn as gnn



confgen = ConformerGeneratorCustom(max_conformers=1,
                                rmsd_threshold=None,
                                force_field="mmff",
                                pool_multiplier=1)

m = Chem.MolFromMolFile('lignin_guaiacyl.mol')

nonring, _ = TorsionFingerprints.CalculateTorsionLists(m)
nr = [list(atoms[0]) for atoms, ang in nonring]
print("nr:", nr)



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


#TODO: Edit class so that it returns a complete representation of edge features
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
        self.observation_space = spaces.Dict({
            'nodes':spaces.Box(low=-np.inf, high=np.inf, shape=(self.mol.GetNumAtoms(), 3)),
            'bonds':spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.bonds), 8)),
            #'angles':spaces.Box(low=-np.inf, high=np.inf, shape=(200, 3)),
            #'dihedrals':spaces.Box(low=-np.inf, high=np.inf, shape=(100, 3)),
            #'num':spaces.Box(low=np.inf, high=np.inf, shape=(4, 1))
        })

        

    def _get_reward(self):
        return np.exp(-1.0 * (confgen.get_conformer_energies(self.mol)[0] - self.standard_energy))

    def _get_obs(self):
        obs = {}
        obs['nodes']=np.array(self.conf.GetPositions())
        
        obs['bonds'] = np.zeros((len(self.bonds), 8))

        for idx, bond in enumerate(self.bonds):
            bt = bond.GetBondType()
            feats = np.array([
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bt == Chem.rdchem.BondType.SINGLE, 
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE, 
                bt == Chem.rdchem.BondType.AROMATIC,
                bond.GetIsConjugated(),
                bond.IsInRing(),
            ])
            obs['bonds'][idx] = feats
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


def obsToGraph(obs):
    positions = obs['nodes'][:obs['num'][0]]
    edge_index1 = obs['bonds'][:obs['num'][1], 0]
    edge_index2 = obs['bonds'][:obs['num'][1], 1]
    edge_index = np.array([np.concatenate((edge_index1,edge_index2)), np.concatenate((edge_index2,edge_index1))])

    edge_attr = np.zeros((obs['num'][1]+obs['num'][2]+obs['num'][3], 8))
    edge_attr[:obs['num'][1], :6] = obs['bonds'][:obs['num'][1], 2:]
    edge_attr = np.concatenate((edge_attr, edge_attr))
    data = Data(
        x=torch.tensor(positions, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        pos=torch.tensor(positions, dtype=torch.float),
    )
    data = Distance()(data)
    return data

def obsToGraphRlpyt(obs):
    #0=nodes/atoms, 1=bonds, 2=angles, 3=dihedrals, 4=num
    if len(obs.nodes) == 1:
        nodes = obs.nodes[0]
    else:
        nodes = obs.nodes
    positions = nodes
    if len(obs.bonds) == 1:
        bonds = obs.bonds[0]
    else:
        bonds = obs.bonds
    edge_index1 = bonds[:, 0]
    edge_index2 = bonds[: , 1]
    edge_index = np.array([np.concatenate((edge_index1,edge_index2)), np.concatenate((edge_index2,edge_index1))])
    edge_attr = bonds[:, 2:]
    edge_attr = np.concatenate((edge_attr, edge_attr))
    data = Data(
        x=torch.tensor(positions, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        pos=torch.tensor(positions, dtype=torch.float),
    )
    data = Distance()(data)
    return data



class ActorNet(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(ActorNet, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(7, dim), nn.ReLU(), nn.Linear(dim, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(5 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, action_dim)

        self.memory = nn.LSTM(2*dim, dim)    
        
        self.action_dim = action_dim
        self.dim = dim
        

    def forward(self, obs, states=None):
        nonring = nr
        data = obs
        data = obsToGraphRlpyt(data)
        data = Batch.from_data_list([data])
        nonring = torch.LongTensor(nonring)
        
        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, 1, self.dim))
            cx = Variable(torch.zeros(1, 1, self.dim))
    
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        pool = self.set2set(out, data.batch)
        lstm_out, (hx, cx) = self.memory(pool.view(1,1,-1), (hx, cx))        
        
        out = torch.index_select(out, dim=0, index=nonring.view(-1))
        out = out.view(4*out.shape[1],-1)
        out = out.permute(1, 0)
        out = torch.cat([out, torch.repeat_interleave(lstm_out, out.shape[0]).view(out.shape[0],-1)], dim=1)
#       
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        
        return out, (hx, cx)       
        
class CriticNet(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(CriticNet, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(7, dim), nn.ReLU(), nn.Linear(dim, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin3 = torch.nn.Linear(dim, 1)
        
        self.action_dim = action_dim
        self.dim = dim
        
        self.memory = nn.LSTM(2*dim, dim)    

    def forward(self, obs, states=None):
        nonring = nr
        data = obs
        data = obsToGraphRlpyt(data)
        data = Batch.from_data_list([data])
        
        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, 1, self.dim))
            cx = Variable(torch.zeros(1, 1, self.dim))
    
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        pool = self.set2set(out, data.batch)
        lstm_out, (hx, cx) = self.memory(pool.view(1,1,-1), (hx, cx))        
        
        out = F.relu(self.lin1(lstm_out.view(1,-1)))
        v = self.lin3(out)
        
        return v, (hx, cx)
        
class RTGN(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(RTGN, self).__init__()
        num_features = 3
        self.action_dim = action_dim
        self.dim = dim
        
        self.actor = ActorNet(action_dim, dim)
        self.critic = CriticNet(action_dim, dim)
        
    def forward(self, obs, states=None):
        
        if states:
            hp, cp, hv, cv = states

            policy_states = (hp, cp)
            value_states = (hv, cv)
        else:
            policy_states = None
            value_states = None
    
        logits, (hp, cp) = self.actor(obs, policy_states)
        v, (hv, cv) = self.critic(obs, value_states)
        
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(0)
        entropy = dist.entropy().unsqueeze(0)

        prediction = {
            'a': action,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }
        pdb.set_trace()
        print("nnet done")
        
        return prediction, (hp, cp, hv, cv)


def make_env():
    return GymEnvWrapper(Environment())


class Mixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape,
        output_size=env_spaces.action.shape)

AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])
class CustomAgent(BaseAgent):
    def __init__(self, ModelCls = RTGN, model_kwargs={"action_dim": 6, "dim": 128}):
        super().__init__(ModelCls=ModelCls, model_kwargs=model_kwargs)

    def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B=global_B, env_ranks=env_ranks)

    def __call__(self, observation, prev_action, prev_reward):
        pred, _ = self.model(observation)
        return pred['a']

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        pred, _ = self.model(observation)
        action = [pred['a']]
        dist_info = DistInfo(prob=pred['log_pi_a'])
        value = pred['v']
        agent_info = AgentInfo(dist_info=dist_info, value=value)
        action, agent_info = buffer_to((action, agent_info), device='cpu')
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        pred, _ = self.model(observation)
        return pred['v']
        




def build_and_train(run_ID=0, cuda_idx=None):
    sampler = SerialSampler(
        EnvCls=make_env,
        env_kwargs={},
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
    )
    algo = PPO()  # Run with defaults.
    agent = CustomAgent()
    runner = MinibatchRl(
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
