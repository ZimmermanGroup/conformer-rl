"""
Molecules
=========
:class:`~conformer_rl.config.mol_config.MolConfig` generators.
"""
from conformer_rl.config import MolConfig
from conformer_rl.utils import calculate_normalizers
from rdkit import Chem
from rdkit.Chem import AllChem

def test_alkane_config() -> MolConfig:
    """Generates a branched alkane :class:`~conformer_rl.config.mol_config.MolConfig` for testing.
    """
    config = config_from_smiles("CC(CCC)CCCC(CCCC)CC", calc_normalizers=False)
    config.E0 = 7.668625034772399
    config.Z0 = 13.263723987526067
    config.tau = 503
    return config

def config_from_molFile(file: str, calc_normalizers: bool=True, ep_steps: int = 200, pruning_thresh: float = 0.05) -> MolConfig:
    """
    """
    mol = Chem.MolFromMolFile(file)
    return config_from_rdkit(mol, calc_normalizers, ep_steps, pruning_thresh)

def config_from_smiles(smiles: str, calc_normalizers: bool=True, ep_steps: int = 200, pruning_thresh: float = 0.05) -> MolConfig:
    """
    """
    mol = Chem.MolFromSmiles(smiles)
    return config_from_rdkit(mol, calc_normalizers, ep_steps, pruning_thresh)

def config_from_rdkit(mol: Chem.rdchem.Mol, calc_normalizers: bool=True, ep_steps: int=200, pruning_thresh: float=0.05) -> MolConfig:
    """
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
