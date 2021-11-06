import conformer_rl
from conformer_rl.logging.env_logger import EnvLogger

def test_basic(mocker):
    logger = EnvLogger(tag = "tag")

    assert logger.tag == "tag"
    assert logger.dir == "data"

    for i in range(10):
        logger.log_step_item("item", i)
        logger.log_step({"doubleitem": i*2, "tripleitem": i*3})

    logger.log_episode_item("epitem", 300)

    assert len(logger.step_data) == 3
    assert len(logger.episode_data) == 1

    logger.log_episode({})

    assert len(logger.episode_data) == 2
    assert "step_data" in logger.episode_data

    logger._add_to_cache(logger.episode_data)
    assert logger.cache["epitem"] == [logger.episode_data["epitem"]]
    assert len(logger.cache) == 2
    assert len(logger.cache["step_data"][0]) == 3

    logger.clear_episode()
    assert len(logger.step_data) == 0
    assert len(logger.episode_data) == 0
    assert len(logger.cache) == 2

    logger.clear_data()
    assert len(logger.cache) == 0

def test_save(mocker):
    logger = EnvLogger(tag = "tag")

    mkd = mocker.patch('conformer_rl.logging.env_logger.mkdir')
    pickle = mocker.patch('conformer_rl.logging.env_logger.pickle')
    chem = mocker.patch('conformer_rl.logging.env_logger.Chem')
    open = mocker.patch('conformer_rl.logging.env_logger.open')

    open.return_value = mocker.Mock()

    molecule = mocker.Mock()
    molecule.GetNumConformers.return_value = 3

    logger.log_episode_item('mol', molecule)

    logger.save_episode(subdir = 'subdir', save_pickle = True, save_molecules = True, save_cache = True)
    
    mkd.assert_called_once_with("data/env_data/tag/subdir")
    assert 'mol' in logger.cache
    assert len(logger.episode_data) == 0

    open.assert_called_with("data/env_data/tag/subdir/data.pickle", "w+b")
    pickle.dump.assert_called_once()
    assert chem.MolToMolFile.call_count == 3



