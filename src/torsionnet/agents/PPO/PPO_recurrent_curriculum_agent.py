import numpy as np
from numpy.lib.financial import npv
import numpy.random
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from collections import deque

from torsionnet.agents.base_agent import BaseAgent, BaseAgentRecurrent
from torsionnet.agents.PPO_recurrent_agent import PPORecurrentAgent
from torsionnet.utils import to_np
from .storage import Storage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPORecurrentCurriculumAgent(PPORecurrentAgent):
    def __init__(self, config):
        super().__init__(config)
        print("running PPO, tag is " + config.tag)
        self.network.to(device)
        self.optimizer = config.optimizer_fn(self.network.parameters()) #optimization function
        self.batch_num = 0

        self.states = self.task.reset()
        with torch.no_grad():
            _, self.recurrent_states = self.network(self.states)
            self.num_recurrent_units = len(self.recurrent_states)

        for state in self.recurrent_states:
            state.zero_()

        self.recurrence = self.config.recurrence

        assert config.rollout_length % self.recurrence == 0
        assert config.mini_batch_size % self.recurrence == 0

        self.curriculum = config.curriculum
        if self.curriculum:
            self.reward_buffer = deque([], maxlen=(config.num_workers + self.curriculum.min_length))
        self.storage = Storage()
        self.total_rewards = np.zeros(config.num_workers)

    def _sample(self):
        config = self.config
        states = self.states
        storage = self.storage
        ##############################################################################################
        #Sampling Loop
        ##############################################################################################
        with torch.no_grad():
            for _ in range(config.rollout_length):
                self.total_steps += config.num_workers

                #add recurrent states (lstm hidden and lstm cell states) to storage
                storage.append({f'recurrent_states_{i}' : rstate for i, rstate in enumerate(self.recurrent_states)})

                #run the neural net once to get prediction
                prediction, self.recurrent_states = self.network(states, self.recurrent_states)

                #step the environment with the action determined by the prediction
                next_states, rewards, terminals, _ = self.task.step(to_np(prediction['a']))

                self.total_rewards += np.asarray(rewards)

                for idx, done in enumerate(terminals):
                    if done:
                        if self.curriculum:
                            self.reward_buffer.append(self.total_rewards[idx])
                        print('logging episodic return train...', self.total_steps)
                        self.train_logger.add_scalar('episodic_return_train', self.total_rewards[idx], self.total_steps)
                        self.total_rewards[idx] = 0.

                        # zero out lstm states for finished environments
                        for rstate in self.recurrent_states:
                            rstate[:, idx].zero_()

                #add everything to storage
                storage.append(prediction)
                storage.append({
                    'states': states,
                    'r': torch.tensor(rewards).unsqueeze(-1).to(device),
                    'm': torch.tensor(1 - terminals).unsqueeze(-1).to(device)
                    })
                states = next_states

            if self.curriculum is not None:
                if len(self.reward_buffer) >= self.curriculum.min_length + config.num_workers:
                    rewbuf = np.array(self.reward_buffer)[-1 * self.curriculum.min_length:]
                    conds = rewbuf > self.curriculum.win_cond

                    if conds.mean() > self.curriculum.success_percent:
                        self.task.env_method('change_level', True)
                        self.reward_buffer.clear()

                    if conds.mean() < self.curriculum.fail_percent:
                        self.task.env_method('change_level', False)
                        self.reward_buffer.clear()

            self.states = states
            prediction, _ = self.network(states, self.recurrent_states)
            self.prediction = prediction
            storage.append(prediction)