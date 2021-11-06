"""
Base_ac_agent_recurrent
=======================
"""
import torch
import numpy as np
from conformer_rl.utils import current_time, load_model, save_model, mkdir, to_np
from conformer_rl.config import Config
import logging
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from conformer_rl.agents.base_ac_agent import BaseACAgent
from conformer_rl.agents.base_agent_recurrent import BaseAgentRecurrent

class BaseACAgentRecurrent(BaseACAgent, BaseAgentRecurrent):
    """Base interface for building reinforcement learning agents that use actor-critic algorithms
    with support for recurrent neural networks"

    Parameters
    ----------
    config : :class:`~conformer_rl.config.agent_config.Config`
        Configuration object for the agent. See notes for a list of config
        parameters used by specific pre-built agents.

    """
    def __init__(self, config: Config):
        super().__init__(config)

        with torch.no_grad():
            _, self.recurrent_states = self.network(self.states)
            self.num_recurrent_units = len(self.recurrent_states)
        for state in self.recurrent_states:
            state.zero_()

        self.recurrence = self.config.recurrence

    def step(self) -> None:
        """Performs one iteration of acquiring samples on the environment
        and then trains on the acquired samples.
        """
        self.storage.reset()
        with torch.no_grad():
            sample_start = time.time()
            self._sample()
            logging.debug(f'sample time: {time.time() - sample_start} seconds')
        train_start = time.time()
        self._calculate_advantages()
        self._train()
        logging.debug(f'train time: {time.time() - train_start} seconds')
        
    def _sample(self) -> None:
        """Collects samples from the training environment.
        """
        config = self.config
        states = self.states
        storage = self.storage
        ##############################################################################################
        #Sampling Loop
        ##############################################################################################
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
                    logging.info(f'logging episodic return train... {self.total_steps}')
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


        self.states = states

        prediction, _ = self.network(states, self.recurrent_states)
        self.prediction = prediction

        storage.append(prediction)