import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable

from deep_rl import *

from deep_rl.component.envs import DummyVecEnv, make_env
from deep_rl.agent.PPO_recurrent_agent_gnn_recurrence import PPORecurrentAgentGnnRecurrence
from deep_rl.agent.PPO_recurrent_agent_gnn import PPORecurrentAgentGnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

HIDDEN_SIZE = 64
env_name = 'CartPole-v0'

torch.manual_seed(1)

class PPORecurrentEvalAgent(PPORecurrentAgentGnnRecurrence):
    def eval_step(self, state, done, rstates):
        with torch.no_grad():
            if done:
                prediction, rstates = self.network(self.config.state_normalizer(state))
            else:
                prediction, rstates = self.network(self.config.state_normalizer(state), rstates)

            out = to_np(prediction['a'])
            return out, rstates
    
    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        done = True
        rstates = None
        while True:
            action, rstates = self.eval_step(state, done, rstates)
            
            done = False
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret   
    
class AdaTask:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5))):
        
        print ("seed is ", seed)
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(name, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.name = name

    def reset(self): 
        print("environment resetting")

        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)
    
class DummyNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)

    def __call__(self, x):
        return x


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 4)
        self.relu = torch.nn.ReLU()

        self.action_memory = nn.LSTM(4, HIDDEN_SIZE)
        self.value_memory = nn.LSTM(4, HIDDEN_SIZE)

        self.action_head = nn.Linear(HIDDEN_SIZE, 2)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)


    def forward(self, x, states = None, actions = None):
        x = tensor(x).unsqueeze(0).to(device)

        if states:
            hp = states[0].detach().to(device)
            cp = states[1].detach().to(device)
            hv = states[2].detach().to(device)
            cv = states[3].detach().to(device)
        else:
            hp = torch.zeros(1, x.shape[0], HIDDEN_SIZE).to(device)
            cp = torch.zeros(1, x.shape[0], HIDDEN_SIZE).to(device)
            hv = torch.zeros(1, x.shape[0], HIDDEN_SIZE).to(device)
            cv = torch.zeros(1, x.shape[0], HIDDEN_SIZE).to(device)
        
        x = self.fc1(x)
        x = self.relu(x)

        action_out, (hp, cp) = self.action_memory(x, (hp, cp))
        value_out, (hv, cv) = self.value_memory(x, (hv, cv))

        v = self.value_head(value_out)
        action_scores = self.action_head(action_out)

        probs = F.softmax(action_scores, dim=-1)
        dist = Categorical(probs)

        if actions != None:
            action = actions
        else:
            action = dist.sample().squeeze(0)

        log_prob = dist.log_prob(action).to(device)
        entropy = dist.entropy().to(device)

        prediction = {
            'a': action,
            'log_pi_a': log_prob.unsqueeze(-1),
            'ent': entropy.unsqueeze(-1),
            'v': v,
        }

        return prediction, (hp, cp, hv, cv)


model = Policy()

def ppo_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    #Constant
    config.network = model
    config.hidden_size = HIDDEN_SIZE
    config.state_normalizer = DummyNormalizer()
    
    #Task
    config.task_fn = lambda: AdaTask(env_name, num_envs = config.num_workers, single_process = False, seed=random.randint(0,7e4))
    config.eval_env = AdaTask(env_name, seed=random.randint(0,7e4))

    #Batch
    config.num_workers = 5
    config.rollout_length = 128 # n_steps
    config.optimization_epochs = 10
    config.mini_batch_size = 32*5
    config.max_steps = 10000000
    config.save_interval = 10000
    config.eval_interval = 2000
    config.eval_episodes = 2
    config.recurrence = 4

    #Coefficients
    lr = 7e-5 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5) #learning_rate #alpha #epsilon
    config.discount = 0.999 # gamma
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0 #ent_coef
    config.gradient_clip = 0.5 #max_grad_norm
    config.ppo_ratio_clip = 0.2


    
    agent = PPORecurrentEvalAgent(config)
    return agent




mkdir('log')
mkdir('tf_log')
set_one_thread()
select_device(0)
tag = "Cartpole-PPO-May8-V0"
agent = ppo_feature(tag=tag)

run_steps(agent)
