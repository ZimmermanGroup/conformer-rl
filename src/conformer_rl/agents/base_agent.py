"""
Base_agent
==========
"""
import torch
import time
import logging
import numpy as np
from conformer_rl.utils import current_time, load_model, save_model, mkdir, to_np
from conformer_rl.logging import TrainLogger, EnvLogger
from conformer_rl.agents.storage import Storage
from conformer_rl.config import Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseAgent:
    """ Base interface for building reinforcement learning agents.

    Parameters
    ----------
    config : :class:`~conformer_rl.config.agent_config.Config`
        Configuration object for the agent. See notes for a list of config
        parameters used by specific pre-built agents.

    Attributes
    ----------
    config : :class:`~conformer_rl.config.agent_config.Config`
        Configuration object used for the agent.
    unique_tag : str
        Unique identifier string for the current training session. Used to identify
        this session in logging.
    eval_logger : :class:`~conformer_rl.logging.env_logger.EnvLogger`
        Used for logging environment data when evaluating agent.
    train_logger :  :class:`~conformer_rl.logging.train_logger.TrainLogger`
        Used for logging agent data while training.
    total_steps : int
        Total number of environment interactions/steps taken by the agent.
    storage : :class:`~conformer_rl.agents.storage.Storage`
        Used to save environment samples, analogous to a replay buffer.

    """
    def __init__(self, config: Config):
        self.config = config
        self.task = config.train_env # gym environment wrapper

        self.dir = config.data_dir
        self.unique_tag = f'{config.tag}_{current_time()}'

        self.eval_logger = EnvLogger(tag=self.unique_tag, dir=self.dir)
        self.train_logger = TrainLogger(tag=self.unique_tag, dir=self.dir, use_tensorboard=config.use_tensorboard, use_cache=False, use_print=False)
        self.total_steps = 0
        self.storage = Storage(config.rollout_length, config.num_workers)

    def run_steps(self) -> None:
        """ Trains the agent.

        Trains the agent until the maximum number of steps (specified by config) is reached.
        Also periodically saves neural network parameters and performs evaluations on the agent,
        if specified in the config.
        """
        config = self.config

        while self.total_steps < config.max_steps:
            if config.save_interval > 0 and self.total_steps % config.save_interval == 0:
                path = self.dir + '/' + 'models' + '/' + self.unique_tag
                mkdir(path)
                self.save(path + '/' +  str(self.total_steps) + '.model')

            if config.eval_interval > 0 and self.total_steps % config.eval_interval == 0:
                eval_start = time.time()
                self.evaluate()
                logging.debug(f'Eval at step {self.total_steps}, eval duration: {time.time() - eval_start} seconds')

            step_start = time.time()
            logging.debug(f'Starting agent step {self.total_steps}')
            self.step()
            logging.debug(f'agent step completed in {time.time() - step_start} seconds')

        self.task.close()

    def step(self) -> None:
        """Performs one iteration of acquiring samples on the environment
        and then trains on the acquired samples.
        """
        raise NotImplementedError

    def _eval_episode(self) -> dict:
        """Evalutes the agent on a single episode of the evaluation environment.

        Returns
        -------
        dict
            Information from the evaluation environment to be logged by the
            `eval_logger`.
        """
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

    def evaluate(self) -> None:
        """Evaluates the agent on the evaluation environment.

        Information dict returned by the environment's :meth:`conformer_rl.environments.conformer_env.ConformerEnv.step` method
        is logged by the `eval_logger` and saved.
        """
        returns = []
        for ep in range(self.config.eval_episodes):
            ep_info = self._eval_episode()
            returns.append(ep_info["total_rewards"])
            
            self.eval_logger.log_episode(ep_info)
            path = f'agent_step_{self.total_steps}' + '/' + f'ep_{ep}'
            self.eval_logger.save_episode(path, save_molecules=True)
            self.train_logger.add_scalar('episodic_return_eval', np.mean(returns), self.total_steps)

    def load(self, filename: str) -> None:
        """Loads the neural network with weights.

        Parameters
        ----------
        filename : str
            The path where the neural network weights are saved.
        """
        load_model(self.network, filename)

    def save(self, filename: str) -> None:
        """ Saves the neural network weights to a file.

        Parameters
        ----------
        filename : str
            The path where the neural network weights are to be saved.
        """
        save_model(self.network, filename)