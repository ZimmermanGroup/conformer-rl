import conformer_rl
from conformer_rl.agents.base_ac_agent_recurrent import BaseAgentRecurrent

def mock_init(self):
    pass

def test_save_evaluate(mocker):
    mocker.patch.object(conformer_rl.agents.base_agent_recurrent.BaseAgentRecurrent, '__init__', mock_init)
    mocker.patch('conformer_rl.agents.base_agent_recurrent.to_np')

    network = mocker.Mock()
    network.return_value = {'a': 'action'}, 'rstates'

    config = mocker.Mock()
    config.eval_episodes = 4
    config.eval_env.reset.return_value = 'reset_state'
    config.eval_env.step.return_value = ('state', 'reward', True, [{'step_info': 'log1', 'episode_info': {'total_rewards': 100}}])

    eval_logger = mocker.Mock()
    train_logger = mocker.Mock()


    agent = BaseAgentRecurrent()
    agent.config = config
    agent.network = network
    agent.eval_logger = eval_logger
    agent.train_logger = train_logger
    agent.total_steps = 59

    info = agent._eval_episode()

    assert(info == {'total_rewards': 100})

    eval_logger.log_step.assert_called_with('log1')
    network.assert_called_with('reset_state', None)
    conformer_rl.agents.base_agent_recurrent.to_np.assert_called_with('action')