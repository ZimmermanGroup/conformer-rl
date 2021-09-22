"""
Base_ac_agent
=============
"""
import torch
import numpy as np
import logging
import time
from conformer_rl.utils import current_time, load_model, save_model, mkdir, to_np
from conformer_rl.config import Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from conformer_rl.agents.base_agent import BaseAgent

class BaseACAgent(BaseAgent):
    """Base interface for building reinforcement learning agents that use actor-critic algorithms.

    Parameters
    ----------
    config : :class:`~conformer_rl.config.agent_config.Config`
        Configuration object for the agent. See notes for a list of config
        parameters used by specific pre-built agents.

    """
    def __init__(self, config: Config):
        super().__init__(config)

        self.network = config.network # neural network / model
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.total_rewards = np.zeros(config.num_workers)
        self.states = self.task.reset()
        self.prediction = None

    def step(self) -> None:
        """Performs one iteration of acquiring samples on the environment
        and then trains on the acquired samples.
        """
        self.storage.reset()
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

            #run the neural net once to get prediction
            prediction = self.network(states)

            #step the environment with the action determined by the prediction
            next_states, rewards, terminals, _ = self.task.step(to_np(prediction['a']))
            self.total_rewards += np.asarray(rewards)

            for idx, done in enumerate(terminals):
                if done:
                    logging.info(f'logging episodic return train... {self.total_steps}')
                    self.train_logger.add_scalar('episodic_return_train', self.total_rewards[idx], self.total_steps)
                    self.total_rewards[idx] = 0.

            #add everything to storage
            storage.append(prediction)
            storage.append({
                'states': states,
                'r': torch.tensor(rewards).unsqueeze(-1).to(device),
                'm': torch.tensor(1 - terminals).unsqueeze(-1).to(device)
                })
            states = next_states


        self.states = states

        prediction = self.network(states)
        self.prediction = prediction

        storage.append(prediction)

    def _train(self) -> None:
        raise NotImplementedError

    def _calculate_advantages(self) -> None:
        """Performs advantage estimation.

        Uses either SARSA or generalized advantage estimation (GAE) for estimating advantages,
        depending on the config.
        """
        config = self.config
        storage = self.storage

        self.advantages, self.returns = [None] * config.rollout_length, [None] * config.rollout_length
        adv = torch.zeros((config.num_workers, 1), dtype=torch.float64).to(device)
        ret = self.prediction['v'].squeeze(0)

        for i in reversed(range(config.rollout_length)):
            ret = storage['r'][i] + config.discount * storage['m'][i] * ret
            if not config.use_gae:
                adv = ret - storage['v'][i].detach()
            else:
                td_error = storage['r'][i] + config.discount * storage['m'][i] * storage['v'][i + 1] - storage['v'][i]
                adv = adv * config.gae_lambda * config.discount * storage['m'][i] + td_error
            self.advantages[i] = adv.detach()
            self.returns[i] = ret.detach()