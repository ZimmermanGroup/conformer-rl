import torch
import numpy as np
from torsionnet.utils import current_time, load_model, save_model, mkdir, to_np
from torsionnet.logging import TrainLogger, EnvLogger
from torsionnet.agents.storage import Storage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.task = config.train_env # gym environment wrapper
        self.network = config.network # neural network / model
        self.optimizer = config.optimizer_fn(self.network.parameters())
        
        self.dir = config.data_dir
        self.unique_tag = f'{config.tag}_{current_time()}'

        self.eval_logger = EnvLogger(tag=self.unique_tag, dir=self.dir)
        self.train_logger = TrainLogger(tag=self.unique_tag, dir=self.dir, use_cache=False, use_print=False)
        self.total_steps = 0
        self.storage = Storage(config.rollout_length, config.num_workers)
        self.total_rewards = np.zeros(config.num_workers)

        self.states = self.task.reset()
        self.prediction = None

    def run_steps(self):
        config = self.config

        while self.total_steps < config.max_steps:
            if config.save_interval > 0 and self.total_steps % config.save_interval == 0:
                path = self.dir + '/' + 'models' + '/' + self.unique_tag
                mkdir(path)
                self.save(path + '/' +  str(self.total_steps) + '.model')

            if config.eval_interval > 0 and self.total_steps % config.eval_interval == 0:
                self.evaluate()

            self.step()

        self.task.close()

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

    def _train(self):
        raise NotImplementedError

    def _eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        info = None
        episode_info = None
        done = False

        with torch.no_grad():
            while not done:
                prediction = self.network(state)
                action = prediction['a']
                state, reward, done, info = env.step(to_np(action))
                self.eval_logger.log_step(info[0]['step_info'])
            return info[0]['episode_info']

    def evaluate(self):
        returns = []
        for ep in range(self.config.eval_episodes):
            ep_info = self._eval_episode()
            returns.append(ep_info["total_rewards"])
            
            self.eval_logger.log_episode(ep_info)
            path = f'agent_step_{self.total_steps}' + '/' + f'ep_{ep}'
            self.eval_logger.save_episode(path, save_molecules=True)
            self.train_logger.add_scalar('episodic_return_eval', np.mean(returns), self.total_steps)

    def load(self, filename: str) -> None:
        load_model(self.network, filename)

    def save(self, filename: str) -> None:
        save_model(self.network, filename)