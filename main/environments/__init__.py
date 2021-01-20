from .environment_wrapper import Task
from .molecule_wrapper import DIFF

from gym.envs.registration import register

register(
    id='ConfEnv-v0',
    entry_point='main.environments.conformer_env:ConformerEnv'
)

