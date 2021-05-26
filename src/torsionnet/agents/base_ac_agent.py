import torch
import numpy as np
from torsionnet.utils import current_time, load_model, save_model, mkdir, to_np
from torsionnet.logging import TrainLogger, EnvLogger
from torsionnet.agents.storage import Storage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torsionnet.agents.base_agent import BaseAgent

class BaseACAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        self.network = config.network # neural network / model
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.total_rewards = np.zeros(config.num_workers)
        self.states = self.task.reset()
        self.prediction = None

    def step(self):
        self.storage.reset()
        self._sample()
        self._calculate_advantages()
        self._train()

    def _sample(self):
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
                    print('logging episodic return train...', self.total_steps)
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

    def _calculate_advantages(self):
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
                adv = adv * config.gae_tau * config.discount * storage['m'][i] + td_error
            self.advantages[i] = adv.detach()
            self.returns[i] = ret.detach()