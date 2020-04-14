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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 64
env_name = 'CartPole-v0'

torch.manual_seed(1)


class A2CRecurrentEvalAgent(A2CRecurrentAgent):
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

class PPORecurrentEvalAgent(PPORecurrentAgent):
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
        self.lstm = nn.LSTMCell(4, HIDDEN_SIZE)
        self.action_head = layer_init(nn.Linear(HIDDEN_SIZE, 2), 1e-3)
        self.value_head = layer_init(nn.Linear(HIDDEN_SIZE, 1), 1e-3)
        self.rewards = []

    def forward(self, x, states=None):
        x = tensor(x)
        x = x.to(device)
        if states:
            if isinstance(states, list):
                h0 = None
                c0 = None
                for state in states:
                    if state[0][0] == None:
                        state = [torch.zeros(len(state[0]), HIDDEN_SIZE), torch.zeros(len(state[1]), HIDDEN_SIZE)]
                    if h0 == None:
                        h0 = state[0]
                        c0 = state[1]
                    else:
                        h0 = torch.cat((h0, state[0]), 0)
                        c0 = torch.cat((c0, state[1]), 0)
                states = h0, c0
            else:
                states = states[0].detach(), states[1].detach()
        else:
            states = Variable(torch.zeros(len(x), HIDDEN_SIZE)), Variable(torch.zeros(len(x), HIDDEN_SIZE))
        states = states[0].to(device), states[1].to(device)
        rstates = self.lstm(x, states)
        x = rstates[0]
        x = x.squeeze(0)

        action_scores = self.action_head(x)
        v = self.value_head(x)

        probs = F.softmax(action_scores, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(0).to(device)
        entropy = dist.entropy().unsqueeze(0).to(device)

        prediction = {
            'a': action,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }

        return prediction, rstates


model = Policy()

def a2c_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 5
    config.task_fn = lambda: AdaTask(env_name, num_envs = config.num_workers, seed=random.randint(0,7e4))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001) #learning_rate #alpha #epsilon
    config.network = model
    config.discount = 0.99 # gamma
    config.use_gae = True
    config.gae_tau = 0.95
    # config.value_loss_weight = 1 # vf_coef
    config.entropy_weight = 0.01 #ent_coef
    config.rollout_length = 5 # n_steps
    config.gradient_clip = 0.5 #max_grad_norm
    config.max_steps = 1000000
    config.save_interval = 10000
    # config.eval_interval = 2000
    # config.eval_episodes = 2
    # config.eval_env = AdaTask(env_name, seed=random.randint(0,7e4))
    config.state_normalizer = DummyNormalizer()
    # config.optimization_epochs = 4
    # config.mini_batch_size = 32

    agent = A2CRecurrentEvalAgent(config)
    return agent

def ppo_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 5
    config.task_fn = lambda: AdaTask(env_name, num_envs = config.num_workers, seed=random.randint(0,7e4))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001) #learning_rate #alpha #epsilon
    config.network = model
    config.discount = 0.99 # gamma
    config.use_gae = True
    config.gae_tau = 0.95
    # config.value_loss_weight = 1 # vf_coef
    config.entropy_weight = 0.01 #ent_coef
    config.rollout_length = 128 # n_steps
    config.gradient_clip = 5 #max_grad_norm
    config.max_steps = 1000000
    config.save_interval = 10000
    # config.eval_interval = 2000
    # config.eval_episodes = 2
    config.eval_env = AdaTask(env_name, seed=random.randint(0,7e4))
    config.state_normalizer = DummyNormalizer()
    config.ppo_ratio_clip = 0.2
    config.optimization_epochs = 10
    config.mini_batch_size = 32*5
    config.recurrence = 1

    agent = PPORecurrentEvalAgent(config)
    return agent




mkdir('log')
mkdir('tf_log')
set_one_thread()
select_device(0)
tag = "TESTING"#'ppo_cartpole_april6_v0'
agent = ppo_feature(tag=tag)

run_steps(agent)
