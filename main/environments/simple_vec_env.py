import numpy as np

class SimpleVecEnv():

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(env_fns)
        self.actions = None

    def step(self, actions):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        return [env.reset() for env in self.envs]

    def close(self):
        for env in self.envs:
            env.close()

    def render(self):
        return [env.render() for env in self.envs]

    def env_method(self, method_name, *method_args, **method_kwargs):
        return [getattr(env, method_name)(*method_args, **method_kwargs) for env in self.envs]
