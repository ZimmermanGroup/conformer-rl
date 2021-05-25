import gym
import numpy as np


# documentation for SubprocVecEnv: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from .simple_vec_env import SimpleVecEnv

from ..utils import mkdir

def make_env(env_id, seed, rank, **kwargs):
    def _thunk():
        np.random.seed(seed + rank)
        env = gym.make(env_id, **kwargs)
        return env

    return _thunk

def Task(name, concurrency=False, num_envs=1, seed=np.random.randint(int(1e5)), **kwargs):
    envs = [make_env(name, seed, i, **kwargs) for i in range(num_envs)]
    return SubprocVecEnv(envs) if concurrency else SimpleVecEnv(envs)
