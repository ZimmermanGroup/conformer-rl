"""
Generate_branched_alkane
========================
"""
from conformer_rl.config import MolConfig
from conformer_rl.molecule_generation.generate_molecule import config_from_rdkit
from rdkit import Chem
import numpy as np
import random

def branched_alkane_config(num_atoms: int) -> MolConfig:
    """Generates a randomized branched alkane :class:`~conformer_rl.config.mol_config.MolConfig`,
    including constants for calculating Gibbs Score.

    Parameters
    ----------
    num_atoms : int
        The number of atoms in the branched alkane.
    """
 
    mol = generate_branched_alkane(num_atoms)
    return config_from_rdkit(mol)

def straight_alkane_config(num_atoms: int) -> MolConfig:
    """Generates a straight alkane chain :class:`~conformer_rl.config.mol_config.MolConfig`,
    including constants for calculating Gibbs Score.

    Parameters
    ----------
    num_atoms : int
        The number of atoms in the alkane.
    """
    mol = Chem.MolFromSmiles('C' * num_atoms)
    return config_from_rdkit(mol)

def generate_branched_alkane(num_atoms: int, save: bool=False) -> Chem.Mol:
    """Generates a branched alkane.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in molecule to be generated.
    save : bool
        Whether to save the molecule as a .mol file.
    """
    mol = Chem.MolFromSmiles('CCCC')
    edit_mol = Chem.RWMol(mol)
    while edit_mol.GetNumAtoms() < num_atoms:
        x = Chem.rdchem.Atom(6)
        randidx = np.random.randint(len(edit_mol.GetAtoms()))
        atom = edit_mol.GetAtomWithIdx(randidx)
        if atom.GetDegree() > 2:
            continue
        if atom.GetDegree() == 2 and random.random() <= 0.5:
            continue
        idx = edit_mol.AddAtom(x)
        edit_mol.AddBond(idx, randidx, Chem.rdchem.BondType.SINGLE)

    Chem.SanitizeMol(edit_mol)
    mol = Chem.rdmolops.AddHs(edit_mol.GetMol())

    if save:
        Chem.rdmolfiles.MolToMolFile(mol, f'{num_atoms}_branched_alkane.mol')
    return mol

