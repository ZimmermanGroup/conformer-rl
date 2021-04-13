from .environment_wrapper import Task
from .conformer_environment import Curriculum

from gym.envs.registration import register

register(
    id='ConfEnv-v0',
    entry_point='torsionnet.environments.environments:GibbsEnv'
)

register(
    id='ConfEnv-v1',
    entry_point='torsionnet.environments.environments:GibbsPruningEnv'
)

register(
    id='XorgateEnv-v0',
    entry_point='torsionnet.environments.environments:XorgateHierarchicalEnv'
)
