import conformer_rl
from conformer_rl.agents.base_agent import BaseAgent
import sys
import pytest


def mock_step(self):
    self.total_steps += 1

def mock_init(self):
    self.total_steps = 0
    self.dir = 'test_dir'
    self.unique_tag = 'unique_tag'


def test_init(mocker):

    unique_tag = 'tag_time'
    tag = 'tag'
    data_dir = 'data'

    mocker.patch('conformer_rl.agents.base_agent.EnvLogger', autospec=True)
    mocker.patch('conformer_rl.agents.base_agent.TrainLogger', autospec=True)
    mocker.patch('conformer_rl.agents.base_agent.Storage', autospec=True)
    mocker.patch('conformer_rl.agents.base_agent.current_time', return_value='time')
    mocker.patch('conformer_rl.agents.base_agent.load_model')
    mocker.patch('conformer_rl.agents.base_agent.save_model')

    config = mocker.Mock()
    config.task
    config.data_dir = data_dir
    config.tag = tag
    config.rollout_length = 7
    config.use_tensorboard=False
    config.train_env.num_envs = 5
    config.network.parameters.return_value = 'params'

    agent = BaseAgent(config)
    conformer_rl.agents.base_agent.Storage.assert_called_with(7, 5)
    conformer_rl.agents.base_agent.EnvLogger.assert_called_with(unique_tag, data_dir)
    config.optimizer_fn.assert_called_with('params')
    conformer_rl.agents.base_agent.TrainLogger.assert_called_with(unique_tag, data_dir, False, False, False)

def test_run_steps(mocker):

    config = mocker.Mock()
    config.save_interval = 2
    config.eval_interval = 2
    config.max_steps = 4

    task = mocker.Mock()
    task.close

    save = mocker.Mock()
    evaluate = mocker.Mock()
    step = mocker.spy(sys.modules[__name__], 'mock_step')

    mocker.patch.object(conformer_rl.agents.base_agent.BaseAgent, '__init__', mock_init)
    mocker.patch.object(conformer_rl.agents.base_agent.BaseAgent, 'save', save)
    mocker.patch.object(conformer_rl.agents.base_agent.BaseAgent, 'evaluate', evaluate)
    mocker.patch.object(conformer_rl.agents.base_agent.BaseAgent, 'step', mock_step)
    mocker.patch('conformer_rl.agents.base_agent.mkdir')

    agent = BaseAgent()
    agent.config = config
    agent.task = task
    agent.run_steps()

    assert(step.call_count == 4)
    assert(save.call_count == 2)
    assert(evaluate.call_count == 2)
    assert(task.close.call_count == 1)
    conformer_rl.agents.base_agent.mkdir.assert_called_with('test_dir/models/unique_tag')

def test_save_load(mocker):
    mocker.patch.object(conformer_rl.agents.base_agent.BaseAgent, '__init__', mock_init)
    mocker.patch('conformer_rl.agents.base_agent.save_model')
    mocker.patch('conformer_rl.agents.base_agent.load_model')

    filename = 'file'
    agent = BaseAgent()
    agent.network = 'network'

    agent.save(filename)
    conformer_rl.agents.base_agent.save_model.assert_called_with(agent.network, filename)
    agent.load(filename)
    conformer_rl.agents.base_agent.load_model.assert_called_with(agent.network, filename)

def test_save_evaluate(mocker):
    mocker.patch.object(conformer_rl.agents.base_agent.BaseAgent, '__init__', mock_init)
    mocker.patch('conformer_rl.agents.base_agent.to_np')

    network = mocker.Mock()
    network.return_value = {'a': 'action'}

    config = mocker.Mock()
    config.eval_episodes = 4
    config.eval_env.reset.return_value = 'reset_state'
    config.eval_env.step.return_value = ('state', 'reward', True, [{'step_info': 'log1', 'episode_info': {'total_rewards': 100}}])

    eval_logger = mocker.Mock()
    train_logger = mocker.Mock()


    agent = BaseAgent()
    agent.config = config
    agent.network = network
    agent.eval_logger = eval_logger
    agent.train_logger = train_logger
    agent.total_steps = 59

    agent.evaluate()

    assert(config.eval_env.reset.call_count == 4)
    network.assert_called_with('reset_state')
    conformer_rl.agents.base_agent.to_np.assert_called_with('action')
    eval_logger.log_episode.assert_called_with({'total_rewards': 100})
    eval_logger.save_episode.assert_called_with('agent_step_59/ep_3', save_molecules=True)
    train_logger.add_scalar.assert_called_with('episodic_return_eval', 100, 59)

def test_unimplemented(mocker):
    mocker.patch.object(conformer_rl.agents.base_agent.BaseAgent, '__init__', mock_init)
    agent = BaseAgent()

    with pytest.raises(NotImplementedError):
        agent.step()
