import conformer_rl
from conformer_rl.agents.base_ac_agent_recurrent import BaseACAgentRecurrent
import numpy as np
import pytest
import torch

def mock_init(self):
    pass

def test_init(mocker):

    def inner_mock_init(self, config):
        self.config = config
        self.states = 'states'
        self.network = mocker.Mock(return_value = ('.', (torch.rand(2, 4), torch.rand(5, 8))))


    mocker.patch('conformer_rl.agents.base_ac_agent_recurrent.BaseACAgent.__init__', inner_mock_init)

    config = mocker.Mock()
    config.recurrence = 10

    agent = BaseACAgentRecurrent(config)
    assert(agent.recurrence == 10)
    assert(agent.num_recurrent_units == 2)
    for state in agent.recurrent_states:
        assert(torch.all(torch.eq(state, torch.zeros(state.shape))))
    
def test_step(mocker):
    storage = mocker.Mock()
    mocker.patch.object(conformer_rl.agents.base_ac_agent_recurrent.BaseACAgentRecurrent, '__init__', mock_init)
    mocker.patch('conformer_rl.agents.base_ac_agent_recurrent.BaseACAgentRecurrent._sample')
    mocker.patch('conformer_rl.agents.base_ac_agent_recurrent.BaseACAgentRecurrent._calculate_advantages')
    mocker.patch('conformer_rl.agents.base_ac_agent_recurrent.BaseACAgentRecurrent._train')

    agent = BaseACAgentRecurrent()
    agent.storage = storage
    agent.step()

    conformer_rl.agents.base_ac_agent_recurrent.BaseACAgentRecurrent._sample.assert_called_once()
    conformer_rl.agents.base_ac_agent_recurrent.BaseACAgentRecurrent._calculate_advantages.assert_called_once()
    conformer_rl.agents.base_ac_agent_recurrent.BaseACAgentRecurrent._train.assert_called_once()
    storage.reset.assert_called_once()

def test_sample(mocker):
    mocker.patch.object(conformer_rl.agents.base_ac_agent_recurrent.BaseACAgentRecurrent, '__init__', mock_init)
    mocker.patch('conformer_rl.agents.base_ac_agent_recurrent.to_np')

    storage = mocker.Mock()

    network = mocker.Mock(return_value = ({'a': 'action'}, ([torch.rand(1, 7, 128), torch.rand(1, 7, 128)])))

    task = mocker.Mock()
    task.step.return_value = ('next_states', [1] * 7, np.array([1, 1, 0, 0, 1, 0, 0]), '.')


    config = mocker.Mock()
    config.rollout_length = 4
    config.num_workers = 7

    train_logger = mocker.Mock()

    agent = BaseACAgentRecurrent()
    agent.storage = storage
    agent.config = config
    agent.total_steps = 0
    agent.states = 'states'
    agent.task = task
    agent.total_rewards = np.array([0] * 7)
    agent.train_logger = train_logger
    agent.network = network
    agent.recurrent_states = ([torch.rand(1, 7, 128), torch.rand(1, 7, 128)])

    agent._sample()

    assert(agent.total_steps == 28)
    assert(np.array_equal(agent.total_rewards, np.array([0, 0, 4, 4, 0, 4, 4])))
    assert(train_logger.add_scalar.call_count == 12)
    print(agent.recurrent_states)
    for idx, done in enumerate([1, 1, 0, 0, 1, 0, 0]):
        if done:
            for i in range(2):
                assert(torch.all(torch.eq(agent.recurrent_states[i][:, idx], torch.zeros(agent.recurrent_states[i][:, idx].shape))))