"""
Base_agent_recurrent
====================
"""
import torch
from conformer_rl.utils import current_time, load_model, save_model, mkdir, to_np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from conformer_rl.agents.base_agent import BaseAgent

from typing import Any, Tuple

class BaseAgentRecurrent(BaseAgent):
    """Base interface for building reinforcement learning agents with support for
    recurrent neural networks.

    Parameters
    ----------
    config : :class:`~conformer_rl.config.agent_config.Config`
        Configuration object for the agent. See notes for a list of config
        parameters used by specific pre-built agents.

    """
    def _eval_step(self, state: object, rstates: Any = None) -> Tuple[Any, Any]:
        """Evalutes the agent on a single step of an episode of the evaluation environment.

        Parameters
        ----------
        state: object
            The current observation from the environment.
        rstates: Any
            Recurrent states from the previous iteration of the neural network.
            If none are supplied, they will be automatically initialized by the
            neural network.

        Returns
        -------
        prediction['a']: Any
            The action to be taken in the next step of the environment.
        rstates: Any
            The next recurrent states to be inputted into the neural network.

        """
        with torch.no_grad():
            prediction, rstates = self.network(state, rstates)

            return prediction['a'], rstates

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
        done = False
        rstates = None
        info = None

        while not done:
            action, rstates = self._eval_step(state, rstates)
            state, reward, done, info = env.step(to_np(action))
            self.eval_logger.log_step(info[0]['step_info'])
        return info[0]['episode_info']
