from rdkit import Chem

def bond_type(bond):
    bt = bond.GetBondType()
    bond_feats = []
    bond_feats = bond_feats + [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return bond_feats

def get_bond_pairs(mol):
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def atom_coords(atom, conf):
    p = conf.GetAtomPosition(atom.GetIdx())
    fts = [p.x, p.y, p.z]
    return fts

def atom_type_CO(atom):
    anum = atom.GetSymbol()
    atom_feats = [
        anum == 'C', anum == 'O',
    ]
    return atom_feats