from conformer_rl.environments import ConformerEnv
from conformer_rl.environments.environment_components.action_mixins import DiscreteActionMixin
from conformer_rl.environments.environment_components.obs_mixins import AtomTypeGraphObsMixin
from conformer_rl.environments.environment_components.reward_mixins import GibbsPruningRewardMixin

import gym
# construct custom environment from pre-built environment mixins
class TestEnv(DiscreteActionMixin, AtomTypeGraphObsMixin, GibbsPruningRewardMixin, ConformerEnv):
    pass

# register the environment with OpenAI gym
gym.register(
    id='TestEnv-v0',
    entry_point='custom_env:TestEnv'
)