Customizing Environment - advanced
==================================

This section will cover an example of creating a custom environment from scratch, 
that is, not using one of the pre-built environment components.

The code for the following examples can be found in
`examples/custom_env_advanced/custom_env.py <https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples/custom_env_advanced>`_.

Example
---------

Suppose we want an environment where conformers are sequentially generated
and the reward for each step of an episode is 1 if the generated conformer has an energy
below some threshold, otherwise the reward is 0.

Overriding the constructor
^^^^^^^^^^^^^^^^^^^^^^^^^^
Since we now have a new parameter for the environment, the energy threshold, we need to override
the constructor for :class:`~conformer_rl.environments.conformer_env.ConformerEnv`. Since the threshold
may differ depending on the molecule, the best way to handle this parameter is to pass it in through the
:class:`~conformer_rl.config.mol_config.MolConfig` object used to initialize the environment::

    class CustomEnv1(ConformerEnv):
        def __init__(self, mol_config: conformer_rl.config.MolConfig, max_steps: int):
            super().__init__(mol_config, max_steps)

            # ensure that mol_config has energy_thresh attribute
            if not hasattr(mol_config,'energy_thresh'):
                raise Exception('mol_config must have energy_thresh attribute to use CustomEnv1')

            # set the energy threshold
            self.energy_thresh = mol_config.energy_thresh
            self.confs_below_threshold = 0

Overriding :meth:`~conformer_rl.environments.conformer_env.ConformerEnv._reward`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, notice that we have initialized another attribute, ``confs_below_threshold``, to
keep track of the number of conformers below the threshold in each episode. Thus, we need to reset this
every episode::

    class CustomEnv1(ConformerEnv):
        def reset(self):
            self.confs_below_threshold = 0
            return super().reset()

Next, we have the overloaded :meth:`~conformer_rl.environments.conformer_env.ConformerEnv._reward` function
return the wanted reward. We also log the energy and the ``confs_below_threshold``::

    class CustomEnv1(ConformerEnv):
        def _reward(self):
            energy = get_conformer_energy(self.mol)
            self.step_info['energy'] = energy # log energy

            reward = 1. if energy < self.energy_thresh else 0.
            if energy < self.energy_thres:
                self.confs_below_threshold += 1
                self.episode_info['confs_below_threshold'] = self.confs_below_threshold
            return reward

Notice that since this class only modifies the reward handler, it can be used with any
observation handler and action handler (include all pre-built ones) as long as the implementation stays independent from the
reward handler.

Finally, to use the environment we must register it with OpenAI gym::

    gym.register(
        id='CustomEnv-v0',
        entry_point='custom_env:CustomEnv'
    )



