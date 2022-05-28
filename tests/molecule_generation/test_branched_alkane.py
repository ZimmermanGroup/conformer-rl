from conformer_rl.molecule_generation.generate_alkanes import generate_branched_alkane


def test_branched_alkane(mocker):
    tofile = mocker.patch('rdkit.Chem.rdmolfiles.MolToMolFile')

    mol = generate_branched_alkane(4)
    assert mol.GetNumAtoms() == 14
    mol = generate_branched_alkane(8)
    assert mol.GetNumAtoms() == 26
    mol = generate_branched_alkane(25)
    assert mol.GetNumAtoms() == 77

    
