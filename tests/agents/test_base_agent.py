import conformer_rl
from conformer_rl.agents.base_agent import BaseAgent

def mock_step(self):
    self.total_steps += 1

def test_base_agent_constructor(mocker):

    unique_tag = 'tag_time'
    tag = 'tag'
    data_dir = 'data'

    mocker.patch('conformer_rl.agents.base_agent.EnvLogger', autospec=True)
    mocker.patch('conformer_rl.agents.base_agent.TrainLogger', autospec=True)
    mocker.patch('conformer_rl.agents.base_agent.Storage', autospec=True)
    mocker.patch('conformer_rl.agents.base_agent.current_time', return_value='time')
    mocker.patch('conformer_rl.agents.base_agent.load_model')
    mocker.patch('conformer_rl.agents.base_agent.save_model')
    mocker.patch.object(conformer_rl.agents.base_agent.BaseAgent, 'step', mock_step)

    config = mocker.Mock()
    config.task
    config.data_dir = data_dir
    config.tag = tag
    config.rollout_length = 7
    config.num_workers = 5
    config.use_tensorboard=False

    agent = BaseAgent(config)
    conformer_rl.agents.base_agent.Storage.assert_called_with(7, 5)
    conformer_rl.agents.base_agent.EnvLogger.assert_called_with(unique_tag, data_dir)
    conformer_rl.agents.base_agent.TrainLogger.assert_called_with(unique_tag, data_dir, False, False, False)

    assert(agent.total_steps == 0)
    agent.step()
    assert(agent.total_steps == 1)


    

