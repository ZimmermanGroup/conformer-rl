"""
Environment_wrapper
===================
"""

import gym
import numpy as np


# documentation for SubprocVecEnv: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from .simple_vec_env import SimpleVecEnv

from typing import Union


def _make_env(env_id, seed, rank, **kwargs):
    def _thunk():
        np.random.seed(seed + rank)
        env = gym.make(env_id, **kwargs)
        return env

    return _thunk

def Task(name: str, concurrency: bool=False, num_envs: int=1, seed: int=np.random.randint(int(1e5)), **kwargs) -> Union[SubprocVecEnv, SimpleVecEnv]:
    """Returns a wrapper for wrapping multiple environments.

    Parameters
    ----------
    name : str
        The name of the environment, as registered using the ``gym.register`` method.
    concurrency : bool
        Whether or not the environments should be run in parallel across multiple CPU's.
    num_envs : bool
        The number of environments to be wrapped.
    seed : bool
        Seed for initializing the environments.

    Returns
    -------
    A wrapper for the environment(s).
    """
    envs = [_make_env(name, seed, i, **kwargs) for i in range(num_envs)]
    return SubprocVecEnv(envs) if concurrency else SimpleVecEnv(envs)