"""
Molecule_features
=================
Helper functions for extracting features from molecules conformers
to generate graph representations in :mod:`conformer_rl.environments.environment_components.obs_mixins`.
"""
from rdkit import Chem
from typing import List

def bond_type(bond: Chem.Bond) -> List[bool]:
    """Extracts features from a bond in a molecule.

    Parameters
    ----------
    bond : rdkit Bond object
        The bond to extract features from

    Returns
    -------
    List of bools of length 6
    Each element corresponds to, respectively, whether the bond is
    
        * A single bond
        * A double bond
        * A triple bond
        * An aromatic bond
        * Is conjugated
        * Is in a ring structure

    """

    bt = bond.GetBondType()
    bond_feats = []
    bond_feats = bond_feats + [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return bond_feats

def get_bond_pairs(mol: Chem.Mol) -> List[List[int]]:
    """Returns a list of all pairs of atoms that have a bond between each other.
    """

    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def atom_coords(atom: Chem.Atom, conf: Chem.Conformer) -> List[float]:
    """Returns the x, y, and z coordinates of an atom in a molecule conformer.
    """
    p = conf.GetAtomPosition(atom.GetIdx())
    fts = [p.x, p.y, p.z]
    return fts

def atom_type_CO(atom: Chem.Atom) -> List[bool]:
    """Returns a one-hot list of length 2 for whether `atom` is a carbon or oxygen atom.
    """
    anum = atom.GetSymbol()
    atom_feats = [
        anum == 'C', anum == 'O',
    ]
    return atom_feats