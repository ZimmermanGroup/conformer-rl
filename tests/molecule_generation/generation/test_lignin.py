from conformer_rl.molecule_generation.generation.generate_lignin import generate_lignin

# additional testing done in jupyter notebook
def test_lignin(mocker):
    mol = generate_lignin(5)


    
