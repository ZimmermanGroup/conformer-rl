from torsionnet.environments.conformer_environment import ConformerEnv
from torsionnet.environments.conformer_environment import GibbsRewardMixin, UniqueGibbsRewardMixin, PruningGibbsRewardMixin
from torsionnet.environments.conformer_environment import DiscreteActionMixin
from torsionnet.environments.conformer_environment import SkeletonPointsObsMixin, XorgateSkeletonPointsObsMixin

class GibbsEnv(GibbsRewardMixin, DiscreteActionMixin, SkeletonPointsObsMixin):
    pass

class GibbsPruningEnv(PruningGibbsRewardMixin, DiscreteActionMixin, SkeletonPointsObsMixin):
    pass

class XorgateHierarchicalEnv(PruningGibbsRewardMixin, DiscreteActionMixin, XorgateSkeletonPointsObsMixin):
    pass