"""
Molecule Config Generators
==========================
Functions for generating :class:`~conformer_rl.config.mol_config.MolConfig` objects given an input molecule.
"""
from conformer_rl.config import MolConfig
from conformer_rl.utils import calculate_normalizers
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import pickle

def test_alkane_config() -> MolConfig:
    config = config_from_smiles("CC(CCC)CCCC(CCCC)CC", num_conformers=200, calc_normalizers=False)
    config.E0 = 7.668625034772399
    config.Z0 = 13.263723987526067
    config.tau = 503
    return config

def config_from_molFile(file: str, num_conformers: int, calc_normalizers: bool = False, pruning_thresh: float = 0.05, save_file: str = "") -> MolConfig:
    """Generates a :class:`~conformer_rl.config.mol_config.MolConfig` object for a molecule specified by the location of a 
    `MOL <https://chem.libretexts.org/Courses/University_of_Arkansas_Little_Rock/ChemInformatics_(2017)%3A_Chem_4399_5399/2.2%3A_Chemical_Representations_on_Computer%3A_Part_II/2.2.2%3A_Anatomy_of_a_MOL_file>`_ file
    containing the molecule.

    Parameters
    ----------
    file : str
        Name of the MOL file containing the molecule to be converted into a :class:`~conformer_rl.config.mol_config.MolConfig` object.

    num_conformers : int
        Number of conformers to be generated. This parameter is also used for calculating normalizers.

    calc_normalizers : bool
        Whether to calculate normalizing constants used in the Gibbs score reward.
        See :class:`~conformer_rl.config.mol_config.MolConfig` for more details.

    pruning_thresh : float
        Torsional fingerprint distance (TFD) threshold for pruning similar conformers when calculating normalizers.
        This parameter is only used for calculating normalizers and is ignored
        if ``calc_normalizers`` is set to ``False``.

    save_file : str
        If not set to an empty string, the generated config object will be saved as a pickle (.pkl) file with the filename
        set to this parameter.

    Returns
    -------
    :class:`~conformer_rl.config.mol_config.MolConfig`
        A :class:`~conformer_rl.config.mol_config.MolConfig` object configured with the input molecule, as well as
        normalizing constants if ``calc_normalizers`` is set to ``True``.

    """
    mol = Chem.MolFromMolFile(file)
    return config_from_rdkit(mol, num_conformers, calc_normalizers, pruning_thresh, save_file)

def config_from_smiles(smiles: str, num_conformers: int, calc_normalizers: bool = False, pruning_thresh: float = 0.05, save_file: str = "") -> MolConfig:
    """Generates a :class:`~conformer_rl.config.mol_config.MolConfig` object for a molecule specified by a 
    `SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_ string.

    Parameters
    ----------
    smiles : str
        A SMILES string representing the molecule.

    num_conformers : int
        Number of conformers to be generated. This parameter is also used for calculating normalizers.

    calc_normalizers : bool
        Whether to calculate normalizing constants used in the Gibbs score reward.
        See :class:`~conformer_rl.config.mol_config.MolConfig` for more details.

    pruning_thresh : float
        Torsional fingerprint distance (TFD) threshold for pruning similar conformers when calculating normalizers.
        This parameter is only used for calculating normalizers and is ignored
        if ``calc_normalizers`` is set to ``False``.

    save_file : str
        If not set to an empty string, the generated config object will be saved as a pickle (.pkl) file with the filename
        set to this parameter.

    Returns
    -------
    :class:`~conformer_rl.config.mol_config.MolConfig`
        A :class:`~conformer_rl.config.mol_config.MolConfig` object configured with the input molecule, as well as
        normalizing constants if ``calc_normalizers`` is set to ``True``.
    """
    mol = Chem.MolFromSmiles(smiles)
    return config_from_rdkit(mol, num_conformers, calc_normalizers, pruning_thresh, save_file)

def config_from_rdkit(mol: Chem.rdchem.Mol, num_conformers: int, calc_normalizers: bool = False, pruning_thresh: float=0.05, save_file: str = "") -> MolConfig:
    """Generates a :class:`~conformer_rl.config.mol_config.MolConfig` object for a molecule specified by an rdkit molecule object.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
        A rdkit molecule object.

    num_conformers : int
        Number of conformers to be generated. This parameter is also used for calculating normalizers.
        
    calc_normalizers : bool
        Whether to calculate normalizing constants used in the Gibbs score reward.
        See :class:`~conformer_rl.config.mol_config.MolConfig` for more details.

    pruning_thresh : float
        Torsional fingerprint distance (TFD) threshold for pruning similar conformers when calculating normalizers.
        This parameter is only used for calculating normalizers and is ignored
        if ``calc_normalizers`` is set to ``False``.

    save_file : str
        If not set to an empty string, the generated config object will be saved as a pickle (.pkl) file with the filename
        set to this parameter.

    Returns
    -------
    :class:`~conformer_rl.config.mol_config.MolConfig`
        A :class:`~conformer_rl.config.mol_config.MolConfig` object configured with the input molecule, as well as
        normalizing constants if ``calc_normalizers`` is set to ``True``.
    """

    config = MolConfig()
    mol = _preprocess_mol(mol)
    config.mol = mol
    config.num_conformers = num_conformers
    if calc_normalizers:
        config.E0, config.Z0 = calculate_normalizers(mol, num_conformers, pruning_thresh)

    logging.info('mol_config object constructed for the following molecule:')
    logging.info(Chem.MolToMolBlock(mol))

    print('\n\nGenerated mol_config attributes: {')
    for key, val in vars(config).items():
        if key != 'mol':
            print(f'\t{key}: {val}')
    print('}')
    print('Please save and reuse above constants when comparing performance on the same task.\n\n')

    if save_file != "":
        if len(save_file) < 4 or save_file[-4:] != '.pkl':
            save_file += '.pkl'
        logging.info(f'saving molecule config as {save_file}')
        with open(save_file, 'w+b') as file:
            pickle.dump(config, file)

    return config

def _preprocess_mol(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    mol = Chem.AddHs(mol)
    AllChem.MMFFSanitizeMolecule(mol)

    return mol
