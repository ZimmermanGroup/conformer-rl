"""
Pre-built Environments
======================

This module contains several pre-built experiments. Each pre-built environment is created by overriding the following components:

* **Action Handler** refers to overriding of the :meth:`~conformer_rl.environments.conformer_env.ConformerEnv._step` method of
  :class:`~conformer_rl.environments.conformer_env.ConformerEnv`, 
  which determines how the molecule is modified given some action.
* **Reward Handler** refers to overriding of the :meth:`~conformer_rl.environments.conformer_env.ConformerEnv._reward` method of
  :class:`~conformer_rl.environments.conformer_env.ConformerEnv`, 
  which determines how the reward is calculated based on the current configuration of the molecule.
* **Observation Handler** refers to overriding of the :meth:`~conformer_rl.environments.conformer_env.ConformerEnv._obs` method of
  :class:`~conformer_rl.environments.conformer_env.ConformerEnv`, 
  which returns an observation object based on the current configuration of the molecule and is a compatible input for the neural net being used for training.

All pre-built environments inherit from :class:`~conformer_rl.environments.conformer_env.ConformerEnv` and share the same constructor.

"""

from conformer_rl.environments.conformer_env import ConformerEnv
from conformer_rl.environments.environment_components.action_mixins import ContinuousActionMixin, DiscreteActionMixin
from conformer_rl.environments.environment_components.reward_mixins import GibbsRewardMixin, GibbsPruningRewardMixin, GibbsEndPruningRewardMixin, GibbsLogPruningRewardMixin
from conformer_rl.environments.environment_components.obs_mixins import GraphObsMixin, AtomCoordsTypeGraphObsMixin



class DiscreteActionEnv(DiscreteActionMixin, GraphObsMixin, ConformerEnv):
    """
    * Action Handler: :class:`~conformer_rl.environments.environment_components.action_mixins.DiscreteActionMixin`
    * Reward Handler: default reward handler from :class:`~conformer_rl.environments.conformer_env.ConformerEnv`
    * Observation Handler: :class:`~conformer_rl.environments.environment_components.obs_mixins.GraphObsMixin`
    """
    pass


class GibbsScoreEnv(GibbsRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, ConformerEnv):
    """
    * Action Handler: :class:`~conformer_rl.environments.environment_components.action_mixins.DiscreteActionMixin`
    * Reward Handler: :class:`~conformer_rl.environments.environment_components.reward_mixins.GibbsRewardMixin`
    * Observation Handler: :class:`~conformer_rl.environments.environment_components.obs_mixins.AtomCoordsTypeGraphObsMixin`
    """
    pass


class GibbsScorePruningEnv(GibbsPruningRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, ConformerEnv):
    """
    * Action Handler: :class:`~conformer_rl.environments.environment_components.action_mixins.DiscreteActionMixin`
    * Reward Handler: :class:`~conformer_rl.environments.environment_components.reward_mixins.GibbsPruningRewardMixin`
    * Observation Handler: :class:`~conformer_rl.environments.environment_components.obs_mixins.AtomCoordsTypeGraphObsMixin`
    """
    pass


class GibbsScoreEndPruningEnv(GibbsEndPruningRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, ConformerEnv):
    """
    * Action Handler: :class:`~conformer_rl.environments.environment_components.action_mixins.DiscreteActionMixin`
    * Reward Handler: :class:`~conformer_rl.environments.environment_components.reward_mixins.GibbsEndPruningRewardMixin`
    * Observation Handler: :class:`~conformer_rl.environments.environment_components.obs_mixins.AtomCoordsTypeGraphObsMixin`
    """
    pass


class GibbsScoreLogPruningEnv(GibbsLogPruningRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, ConformerEnv):
    """
    * Action Handler: :class:`~conformer_rl.environments.environment_components.action_mixins.DiscreteActionMixin`
    * Reward Handler: :class:`~conformer_rl.environments.environment_components.reward_mixins.GibbsLogPruningRewardMixin`
    * Observation Handler: :class:`~conformer_rl.environments.environment_components.obs_mixins.AtomCoordsTypeGraphObsMixin`
    """
    pass