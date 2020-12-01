#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave
import pickle
import time

def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
        if config.log_interval and not agent.total_steps % config.log_interval:
            # agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        # agent.switch_task()


class BaseAgent:
    def __init__(self, config):
        self.config = config
        # self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def eval_step(self, state):
        raise NotImplementedError

    def eval_episode(self):
        raise NotImplementedError

    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        # self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
        #     self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        # ))
        # self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        # return {
        #     'episodic_return_test': np.mean(episodic_returns),
        # }

    # def record_online_return(self, info, offset=0):
    #     if isinstance(info, dict):
    #         ret = info['episodic_return']
    #         if ret is not None:
    #             self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
    #             self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))

    #         for key in info:
    #             if key == 'episodic_return' or key == 'terminal_observation':
    #                 continue
    #             else:
    #                 if key and info[key]:
    #                     self.logger.add_scalar(key, info[key], self.total_steps + offset)

    #     elif isinstance(info, tuple):
    #         for i, info_ in enumerate(info):
    #             self.record_online_return(info_, i)
    #     else:
    #         raise NotImplementedError

    # def switch_task(self):
    #     config = self.config
    #     if not config.tasks:
    #         return
    #     segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
    #     if self.total_steps > segs[self.task_ind + 1]:
    #         self.task_ind += 1
    #         self.task = config.tasks[self.task_ind]
    #         self.states = self.task.reset()
    #         self.states = config.state_normalizer(self.states)
    #         self.done = True

    # def record_episode(self, dir, env):
    #     mkdir(dir)
    #     steps = 0
    #     state = env.reset()
    #     while True:
    #         self.record_obs(env, dir, steps)
    #         action = self.record_step(state)
    #         state, reward, done, info = env.step(action)
    #         ret = info[0]['episodic_return']
    #         steps += 1
    #         if ret is not None:
    #             break

    # def record_step(self, state):
    #     raise NotImplementedError

    # # For DMControl
    # def record_obs(self, env, dir, steps):
    #     env = env.env.envs[0]
    #     obs = env.render(mode='rgb_array')
    #     imsave('%s/%04d.png' % (dir, steps), obs)
