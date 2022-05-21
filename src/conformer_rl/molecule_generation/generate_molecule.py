"""
Molecule Generators
===================
Functions for generating :class:`~conformer_rl.config.mol_config.MolConfig` objects given an input molecule.
"""
from conformer_rl.config import MolConfig
from conformer_rl.utils import calculate_normalizers
from rdkit import Chem
from rdkit.Chem import AllChem

def test_alkane_config() -> MolConfig:
    config = config_from_smiles("CC(CCC)CCCC(CCCC)CC", calc_normalizers=False)
    config.E0 = 7.668625034772399
    config.Z0 = 13.263723987526067
    config.tau = 503
    return config

def config_from_molFile(file: str, calc_normalizers: bool=True, ep_steps: int = 200, pruning_thresh: float = 0.05) -> MolConfig:
    """Generates a :class:`~conformer_rl.config.mol_config.MolConfig` object for a molecule specified by the location of a 
    `MOL <https://chem.libretexts.org/Courses/University_of_Arkansas_Little_Rock/ChemInformatics_(2017)%3A_Chem_4399_5399/2.2%3A_Chemical_Representations_on_Computer%3A_Part_II/2.2.2%3A_Anatomy_of_a_MOL_file>`_ file
    containing the molecule.

    Parameters
    ----------
    file : str
        Name of the MOL file containing the molecule to be converted into a :class:`~conformer_rl.config.mol_config.MolConfig` object.

    calc_normalizers : bool
        Whether to calculate normalizing constants used in the Gibbs score reward.
        See :class:`~conformer_rl.config.mol_config.MolConfig` for more details.

    ep_steps : int
        Number of conformers to be generated. This parameter is only used for calculating normalizers and is ignored
        if ``calc_normalizers`` is set to ``False``.

    pruning_thresh : float
        Torsional fingerprint distance (TFD) threshold for pruning similar conformers when calculating normalizers.
        This parameter is only used for calculating normalizers and is ignored
        if ``calc_normalizers`` is set to ``False``.

    Returns
    -------
    :class:`~conformer_rl.config.mol_config.MolConfig`
        A :class:`~conformer_rl.config.mol_config.MolConfig` object configured with the input molecule, as well as
        normalizing constants if ``calc_normalizers`` is set to ``True``.

    """
    mol = Chem.MolFromMolFile(file)
    return config_from_rdkit(mol, calc_normalizers, ep_steps, pruning_thresh)

def config_from_smiles(smiles: str, calc_normalizers: bool=True, ep_steps: int = 200, pruning_thresh: float = 0.05) -> MolConfig:
    """Generates a :class:`~conformer_rl.config.mol_config.MolConfig` object for a molecule specified by a 
    `SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_ string.

    Parameters
    ----------
    smiles : str
        A SMIELS string representing the molecule.

    calc_normalizers : bool
        Whether to calculate normalizing constants used in the Gibbs score reward.
        See :class:`~conformer_rl.config.mol_config.MolConfig` for more details.

    ep_steps : int
        Number of conformers to be generated. This parameter is only used for calculating normalizers and is ignored
        if ``calc_normalizers`` is set to ``False``.

    pruning_thresh : float
        Torsional fingerprint distance (TFD) threshold for pruning similar conformers when calculating normalizers.
        This parameter is only used for calculating normalizers and is ignored
        if ``calc_normalizers`` is set to ``False``.

    Returns
    -------
    :class:`~conformer_rl.config.mol_config.MolConfig`
        A :class:`~conformer_rl.config.mol_config.MolConfig` object configured with the input molecule, as well as
        normalizing constants if ``calc_normalizers`` is set to ``True``.
    """
    mol = Chem.MolFromSmiles(smiles)
    return config_from_rdkit(mol, calc_normalizers, ep_steps, pruning_thresh)

def config_from_rdkit(mol: Chem.rdchem.Mol, calc_normalizers: bool=True, ep_steps: int=200, pruning_thresh: float=0.05) -> MolConfig:
    """Generates a :class:`~conformer_rl.config.mol_config.MolConfig` object for a molecule specified by an rdkit molecule object.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
        A rdkit molecule object.

    calc_normalizers : bool
        Whether to calculate normalizing constants used in the Gibbs score reward.
        See :class:`~conformer_rl.config.mol_config.MolConfig` for more details.

    ep_steps : int
        Number of conformers to be generated. This parameter is only used for calculating normalizers and is ignored
        if ``calc_normalizers`` is set to ``False``.

    pruning_thresh : float
        Torsional fingerprint distance (TFD) threshold for pruning similar conformers when calculating normalizers.
        This parameter is only used for calculating normalizers and is ignored
        if ``calc_normalizers`` is set to ``False``.

    Returns
    -------
    :class:`~conformer_rl.config.mol_config.MolConfig`
        A :class:`~conformer_rl.config.mol_config.MolConfig` object configured with the input molecule, as well as
        normalizing constants if ``calc_normalizers`` is set to ``True``.
    """

    config = MolConfig()
    mol = _preprocess_mol(mol)
    config.mol = mol
    if calc_normalizers:
        config.E0, config.Z0 = calculate_normalizers(mol, ep_steps, pruning_thresh)
    return config

def _preprocess_mol(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    mol = Chem.AddHs(mol)
    AllChem.MMFFSanitizeMolecule(mol)

    return mol
