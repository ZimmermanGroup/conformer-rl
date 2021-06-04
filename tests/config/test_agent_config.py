from conformer_rl.config.agent_config import Config

def test_config():
    config = Config()
    assert config.tag == 'test'