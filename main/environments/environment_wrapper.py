import gym
import numpy as np
import torch
import os


# documentation for SubprocVecEnv: https://stable-baselines.readthedocs.io/en/v2.5.0/guide/vec_envs.html
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from ..utils import mkdir, random_seed

def make_env(env_id, seed, rank):
    def _thunk():
        # random_seed(seed + rank)
        env = gym.make(env_id)
        env = OriginalReturnWrapper(env)
        return env

    return _thunk


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

def Task(name, num_envs=1, seed=np.random.randint(int(1e5))):
    envs = [make_env(name, seed, i) for i in range(num_envs)]
    return SubprocVecEnv(envs)
