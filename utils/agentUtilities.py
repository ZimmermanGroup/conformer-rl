import gym
from gym import spaces
from gym.envs.registration import registry, register, make, spec
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import py3Dmol
from deep_rl import *
from deep_rl.component.envs import DummyVecEnv


class A2CEvalAgent(A2CAgent):
    def eval_step(self, state):
        prediction = self.network(self.config.state_normalizer(state))
        return prediction['a']

class A2CRecurrentEvalAgent(A2CRecurrentAgent):
    def eval_step(self, state, done, rstates):
        with torch.no_grad():
            if done:
                prediction, rstates = self.network(self.config.state_normalizer(state))
            else:
                prediction, rstates = self.network(self.config.state_normalizer(state), rstates)

            return prediction['a'], rstates

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        done = True
        rstates = None
        while True:
            action, rstates = self.eval_step(state, done, rstates)
            done = False
            state, reward, done, info = env.step(to_np(action))
            ret = info[0]['episodic_return']
            if ret is not None:
                break

        return ret

class PPORecurrentEvalAgent(PPORecurrentAgentRecurrence):
    def eval_step(self, state, done, rstates):
        with torch.no_grad():
            if done:
                prediction, rstates = self.network(self.config.state_normalizer(state))
            else:
                prediction, rstates = self.network(self.config.state_normalizer(state), rstates)

            return prediction['a'], rstates

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        done = True
        rstates = None
        while True:
            action, rstates = self.eval_step(state, done, rstates)
            done = False
            state, reward, done, info = env.step(to_np(action))
            ret = info[0]['episodic_return']
            if ret is not None:
                break

        return ret


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

    # def change_level(self, level):
    #     return self.env.change_level(level)

    def reset(self):
        return self.env.reset()


def make_env(env_id, seed, rank, episode_life=True):
    def _thunk():
        random_seed(seed + rank)
        env = gym.make(env_id)
        env = OriginalReturnWrapper(env)
        return env

    return _thunk


class AdaTask:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5))):
        if log_dir is not None:
            mkdir(log_dir)

        logging.info(f'seed is {seed}')

        envs = [make_env(name, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
            self.env = DummyVecEnv(envs)
        else:
            self.env = SubprocVecEnv(envs)
        self.name = name

    def change_level(self, xyz):
        self.env_method('change_level', xyz)

    def env_method(self, method_name, xyz):
        return self.env.env_method(method_name, xyz)

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

class DummyNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)

    def __call__(self, x):
        return x
