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

        self.dir = config.data_dir
        self.unique_tag = f'{config.tag}_{current_time()}'

        self.eval_logger = EnvLogger(tag=self.unique_tag, dir=self.dir)
        self.train_logger = TrainLogger(tag=self.unique_tag, dir=self.dir, use_cache=False, use_print=False)
        self.total_steps = 0
        self.storage = Storage(config.rollout_length, config.num_workers)

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
        raise NotImplementedError

    def _train(self):
        raise NotImplementedError

    def _eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        info = None
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