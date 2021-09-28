"""
Molecules
=========
:class:`~conformer_rl.config.mol_config.MolConfig` generators.
"""
from conformer_rl.config import MolConfig
from conformer_rl.molecule_generation import generation
from conformer_rl.utils import calculate_normalizers
from rdkit import Chem
from rdkit.Chem import AllChem

def branched_alkane(num_atoms: int) -> MolConfig:
    """Generates a randomized branched alkane :class:`~conformer_rl.config.mol_config.MolConfig`,
    including constants for calculating Gibbs Score.

    Parameters
    ----------
    num_atoms : int
        The number of atoms in the branched alkane.
    """
    config = MolConfig()

    mol = generation.generate_branched_alkane(num_atoms)
    mol = Chem.AddHs(mol)
    AllChem.MMFFSanitizeMolecule(mol)
    E0, Z0 = calculate_normalizers(mol)
    
    config.mol = mol
    config.E0 = E0
    config.Z0 = Z0
    return config

def straight_alkane(num_atoms: int) -> MolConfig:
    """Generates a straight alkane chain :class:`~conformer_rl.config.mol_config.MolConfig`,
    including constants for calculating Gibbs Score.

    Parameters
    ----------
    num_atoms : int
        The number of atoms in the alkane.
    """
    config = MolConfig()

    mol = Chem.MolFromSmiles('C' * num_atoms)
    mol = Chem.AddHs(mol)
    AllChem.MMFFSanitizeMolecule(mol)
    E0, Z0 = calculate_normalizers(mol)

    config.mol = mol
    config.E0 = E0
    config.Z0 = Z0
    return config

def lignin(num_monomers: int) -> MolConfig:
    """Generates a lignin :class:`~conformer_rl.config.mol_config.MolConfig`,
    including constants for calculating Gibbs Score.

    Parameters
    ----------
    num_monomers : int
        Number of monomers in the lignin.
    """
    config = MolConfig()

    mol = generation.generate_lignin(num_monomers)
    mol = Chem.AddHs(mol)
    AllChem.MMFFSanitizeMolecule(mol)
    E0, Z0 = calculate_normalizers(mol)

    config.mol = mol
    config.E0 = E0
    config.Z0 = Z0
    return config

def xorgate(gate_complexity: int, num_gates: int) -> MolConfig:
    """Generates a xorgate :class:`~conformer_rl.config.mol_config.MolConfig`.

    An xorgate is a chain of alternating gates, where each gate is a chain of benzenes
    with a single carbon chain tail. 

    Parameters
    ----------
    gate_complexity : int
        Number of benzenes in each gate.
    num_gates : int
        Number of gates in molecule.
    """

    config = MolConfig()
    mol = generation.generate_xor_gate(gate_complexity, num_gates)
    mol = Chem.AddHs(mol)
    AllChem.MMFFSanitizeMolecule(mol)
    E0, Z0 = calculate_normalizers(mol)

    config.mol = mol
    config.E0 = E0
    config.Z0 = Z0
    return config



def test_alkane() -> MolConfig:
    """Generates a branched alkane :class:`~conformer_rl.config.mol_config.MolConfig` for testing.
    """
    config = MolConfig()
    
    mol = Chem.MolFromSmiles("CC(CCC)CCCC(CCCC)CC")
    mol = Chem.AddHs(mol)
    AllChem.MMFFSanitizeMolecule(mol)

    config.mol = mol
    config.E0 = 7.668625034772399
    config.Z0 = 13.263723987526067
    config.tau = 503
    return config

def mol_from_molFile(file: str, ep_steps: int = 200, pruning_thresh: float = 0.05) -> MolConfig:
    """
    """
    config = MolConfig()

    mol = Chem.MolFromMolFile(file)
    mol = _preprocess_mol(mol)
    config.mol = mol
    config.E0, config.Z0 = calculate_normalizers(mol, ep_steps, pruning_thresh)
    return config

def mol_from_smiles(smiles: str, ep_steps: int = 200, pruning_thresh: float  = 0.05) -> MolConfig:
    """
    """
    config = MolConfig()

    mol = Chem.MolFromSmiles(smiles)
    mol = _preprocess_mol(mol)
    config.mol = mol
    config.E0, config.Z0 = calculate_normalizers(mol, ep_steps, pruning_thresh)
    return config


def _preprocess_mol(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    mol = Chem.AddHs(mol)
    AllChem.MMFFSanitizeMolecule(mol)

    return mol
