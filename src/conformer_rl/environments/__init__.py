from .environment_wrapper import Task
from .conformer_env import ConformerEnv
import inspect

from conformer_rl.environments import environments

from gym.envs.registration import register

envs = [m[0] for m in inspect.getmembers(environments, inspect.isclass) if m[1].__module__ == 'conformer_rl.environments.environments']
for env in envs:
    register(
        id = env + '-v0',
        entry_point = 'conformer_rl.environments.environments:' + env
    )