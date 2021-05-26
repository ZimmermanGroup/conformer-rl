import torch
import numpy as np
from torsionnet.utils import current_time, load_model, save_model, mkdir, to_np
from torsionnet.logging import TrainLogger, EnvLogger
from torsionnet.agents.storage import Storage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torsionnet.agents.base_agent import BaseAgent

class BaseAgentRecurrent(BaseAgent):
    def _eval_step(self, state, done, rstates):
        with torch.no_grad():
            if done:
                prediction, rstates = self.network(state)
            else:
                prediction, rstates = self.network(state, rstates)

            return prediction['a'], rstates

    def _eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        done = False
        rstates = None
        info = None

        while not done:
            action, rstates = self._eval_step(state, done, rstates)
            state, reward, done, info = env.step(to_np(action))
            self.eval_logger.log_step(info[0]['step_info'])
        return info[0]['episode_info']
