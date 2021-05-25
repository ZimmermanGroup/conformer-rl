from torsionnet.environments.conformer_env import ConformerEnv
from torsionnet.environments.environment_components.action_mixins import ContinuousActionMixin, DiscreteActionMixin
from torsionnet.environments.environment_components.reward_mixins import GibbsRewardMixin, GibbsPruningRewardMixin, GibbsEndPruningRewardMixin, GibbsLogPruningRewardMixin
from torsionnet.environments.environment_components.obs_mixins import GraphObsMixin, AtomCoordsTypeGraphObsMixin



class DiscreteActionEnv(DiscreteActionMixin, GraphObsMixin, ConformerEnv):
    pass


class GibbsScoreEnv(GibbsRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, ConformerEnv):
    pass


class GibbsScorePruningEnv(GibbsPruningRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, ConformerEnv):
    pass


class GibbsScoreEndPruningEnv(GibbsEndPruningRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, ConformerEnv):
    pass


class GibbsScoreLogPruningEnv(GibbsLogPruningRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, ConformerEnv):
    pass