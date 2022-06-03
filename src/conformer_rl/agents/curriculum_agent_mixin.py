import logging
import time
from collections import deque

import numpy as np
import torch

class ExternalCurriculumAgentMixin():
    def __init__(self, config):
        super().__init__(config)
        self.reward_buffer = deque([], maxlen=config.curriculum_agent_buffer_len)
        self.curriculum_buffer_len = config.curriculum_agent_buffer_len
        self.curriculum_reward_thresh = config.curriculum_agent_reward_thresh
        self.curriculum_success_rate = config.curriculum_agent_success_rate
        self.curriculum_fail_rate = config.curriculum_agent_fail_rate

    def step(self) -> None:

        # sample 
        self.storage.reset()
        sample_start = time.time()
        self._sample()
        logging.debug(f'sample time: {time.time() - sample_start} seconds')

        # update curriculum
        self.update_curriculum()

        # train
        train_start = time.time()
        self._calculate_advantages()
        self._train()
        logging.debug(f'train time: {time.time() - train_start} seconds')

    def update_curriculum(self) -> None:
        current_terminals = torch.cat(self.storage['terminals']).squeeze()
        current_rewards = torch.cat(self.storage['r']).squeeze()
        self.reward_buffer.extend(current_rewards[current_terminals == True].tolist())

        if len(self.reward_buffer) >= self.curriculum_buffer_len:
            rewbuf = np.array(self.reward_buffer)
            pass_rate = (rewbuf >= self.curriculum_reward_thresh).mean()

            if pass_rate > self.curriculum_success_rate:
                self.task.env_method('increase_level')
                self.reward_buffer.clear()
            elif pass_rate < self.curriculum_fail_rate:
                self.task.env_method('decrease_level')
                self.reward_buffer.clear()