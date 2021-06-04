from conformer_rl.config.mol_config import MolConfig

def test_config():
    config = MolConfig()
    assert config.mol is None