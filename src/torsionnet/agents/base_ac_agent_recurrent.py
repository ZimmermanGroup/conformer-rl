import torch
import numpy as np
from torsionnet.utils import current_time, load_model, save_model, mkdir, to_np
from torsionnet.logging import TrainLogger, EnvLogger
from torsionnet.agents.storage import Storage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torsionnet.agents.base_ac_agent import BaseACAgent
from torsionnet.agents.base_agent_recurrent import BaseAgentRecurrent

class BaseACAgentRecurrent(BaseACAgent, BaseAgentRecurrent):
    def __init__(self, config):
        super().__init__(config)

        with torch.no_grad():
            _, self.recurrent_states = self.network(self.states)
            self.num_recurrent_units = len(self.recurrent_states)
        for state in self.recurrent_states:
            state.zero_()

        self.recurrence = self.config.recurrence

    def step(self):
        self.storage.reset()
        with torch.no_grad():
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

                #add recurrent states (lstm hidden and lstm cell states) to storage
                storage.append({f'recurrent_states_{i}' : rstate for i, rstate in enumerate(self.recurrent_states)})

                #run the neural net once to get prediction
                prediction, self.recurrent_states = self.network(states, self.recurrent_states)

                #step the environment with the action determined by the prediction
                next_states, rewards, terminals, _ = self.task.step(to_np(prediction['a']))

                self.total_rewards += np.asarray(rewards)

                for idx, done in enumerate(terminals):
                    if done:
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


            self.states = states

            prediction, _ = self.network(states, self.recurrent_states)
            self.prediction = prediction

            storage.append(prediction)