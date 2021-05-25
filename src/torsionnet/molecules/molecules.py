from torsionnet.config import MolConfig
from torsionnet.molecules import generate_molecule
from torsionnet.utils import calculate_normalizers
from rdkit import Chem

def branched_alkane(num_atoms: int) -> MolConfig:
    config = MolConfig()

    mol = generate_molecule.generate_branched_alkane(num_atoms)
    mol = Chem.AddHs(mol)
    E0, Z0 = calculate_normalizers(mol)
    
    config.mol = mol
    config.E0 = E0
    config.Z0 = Z0
    return config

def straight_alkane(num_atoms: int) -> MolConfig:
    config = MolConfig()

    mol = Chem.MolFromSmiles('C' * num_atoms)
    mol = Chem.AddHs(mol)
    E0, Z0 = calculate_normalizers(mol)

    config.mol = mol
    config.E0 = E0
    config.Z0 = Z0
    return config

def lignin(num_monomers: int) -> MolConfig:
    config = MolConfig()

    mol = generate_molecule.generate_lignin(num_monomers)
    mol = Chem.AddHs(mol)
    E0, Z0 = calculate_normalizers(mol)

    config.mol = mol
    config.E0 = E0
    config.Z0 = Z0
    return config

def test_alkane() -> MolConfig:
    config = MolConfig()
    
    mol = Chem.MolFromSmiles("CC(CCC)CCCC(CCCC)CC")
    mol = Chem.AddHs(mol)

    config.mol = mol
    config.E0 = 7.668625034772399
    config.Z0 = 13.263723987526067
    config.tau = 503
    return config

def mol_from_dict(input: dict) -> MolConfig:
    config = MolConfig()

    mol = input['mol']
    mol = Chem.AddHs(mol)
    config.mol = mol

    for key, val in input.items():
        if key != 'mol':
            setattr(config, key, val)
    return config