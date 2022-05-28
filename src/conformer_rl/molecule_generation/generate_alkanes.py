"""
Alkane Generators
=================
"""
from rdkit import Chem
import numpy as np
import random

def generate_straight_alkane(num_atoms: int) -> Chem.Mol:
    """Generates a straight alkane chain
    
    Parameters
    ----------
    num_atoms : int
        Number of atoms in the molecule."""

    return Chem.MolFromSmiles('C' * num_atoms)

def generate_branched_alkane(num_atoms: int) -> Chem.Mol:
    """Generates a branched alkane.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in molecule to be generated.
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

    return mol

