import pytest
from .eval_logger import EvalLogger

@pytest.fixture
def logger():
    ''' return eval logger object logging to current directory '''
    return EvalLogger('.')

@pytest.fixture
def step_data_1():
    return {
        "testint": 1,
        "teststring": 'test',
        "testfloat": 0.142,
    }

@pytest.fixture
def step_data_2():
    return {
        "testint": 12,
        "teststring": 'test2',
        "testfloat": 0.152,
    }

@pytest.fixture
def episode_data_1():
    return {
        "episodedata": 123
    }

def test_log_step(logger, step_data_1):
    assert logger.step_data == {}
    assert logger.episode_data == {}

    logger.log_step(step_data_1)
    assert logger.step_data["testint"] == [1]
    assert logger.step_data["teststring"] == ['test']
    assert logger.step_data["testfloat"] == [0.142]

def test_log_step2(logger, step_data_1, step_data_2):
    logger.log_step(step_data_1)
    logger.log_step(step_data_2)
    assert logger.step_data["testint"] == [1, 12]
    assert logger.step_data["teststring"] == ['test', 'test2']
    assert logger.step_data["testfloat"] == [0.142, 0.152]

def test_log_episode(logger, step_data_1, episode_data_1):
    logger.log_step(step_data_1)
    logger.log_episode(episode_data_1)
    assert logger.step_data["testint"] == [1]
    assert logger.step_data["teststring"] == ['test']
    assert logger.step_data["testfloat"] == [0.142]
    assert logger.episode_data["step_data"]["testint"] == [1]
    assert logger.episode_data["step_data"]["teststring"] == ['test']
    assert logger.episode_data["step_data"]["testfloat"] == [0.142]
    assert logger.episode_data["episodedata"] == 123

def test_dump_episode(logger, step_data_1, episode_data_1):
    agent_steps = 0
    logger.log_step(step_data_1)
    logger.log_episode(episode_data_1)
    logger.dump_episode(agent_steps)
    assert logger.step_data == {}
    assert logger.episode_data == {}

