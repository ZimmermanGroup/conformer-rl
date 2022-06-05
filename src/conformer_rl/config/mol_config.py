"""
Mol_config
==========
"""

class MolConfig:
    """Configuration object for environments.

    Specifies parameters for :class:`~conformer_rl.environments.conformer_env.ConformerEnv`
    conformer environments.

    Attributes
    ----------
    mol : rdkit Mol, required for all environments
        The molecule to be used by the environment.
    num_conformers : int
        The number of conformers to be generated in each episode of the environment for this molecule. In other words,
        the number of steps the environment will iterate through in each episode before resetting and entering a new episode.
    seed: int, required for all environments
        Seed for generating initial conformers for the molecule. If set to -1,
        the seed is randomized.
    E0: float, required for environments that use Gibbs score reward.
        The normalizing :math:`E_0` parameter for calculating Gibbs Score for the molecule. See [1]_ for more details.
    Z0 : float, required for environments that use Gibbs score reward.
        The normalizing :math:`Z_0` parameter for calculating Gibbs score for the molecule. See [1]_ for more details.
    tau : float, required for environments that use Gibbs score reward.
        The temperature (τ) parameter for calculating Gibbs Score for the molecule. See [1]_ for more details.

    pruning_thresh : float, required for environments that use pruning.
        The minimum allowed TFD (torsional fingerprint deviation) between conformers when pruning.

    References
    ----------
    .. [1] `TorsionNet paper <https://arxiv.org/abs/2006.07078>`_
    
    """

    def __init__(self):
        self.mol = None
        self.num_conformers = 200
        self.seed = -1

        # Parameters for using Gibbs Score
        self.E0 = 0
        self.Z0 = 1
        self.tau = 300

        # Parameters used for pruning 
        self.pruning_thresh = 0.05
