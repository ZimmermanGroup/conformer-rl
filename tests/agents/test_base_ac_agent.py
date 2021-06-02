import conformer_rl
from conformer_rl.agents.base_ac_agent import BaseACAgent
import numpy as np
import pytest
import torch

def mock_init(self):
    pass

def test_init(mocker):

    def inner_mock_init(self, config):
        self.task = mocker.Mock()

    mocker.patch('conformer_rl.agents.base_ac_agent.BaseAgent.__init__', inner_mock_init)
    mocker.patch('conformer_rl.agents.base_ac_agent.np.zeros')

    config = mocker.Mock()
    config.num_workers = 5
    config.network.parameters.return_value = 'params'

    agent = BaseACAgent(config)
    conformer_rl.agents.base_ac_agent.np.zeros.assert_called_with(5)
    config.optimizer_fn.assert_called_with('params')
    agent.task.reset.assert_called()
    
def test_step(mocker):
    storage = mocker.Mock()
    mocker.patch.object(conformer_rl.agents.base_ac_agent.BaseACAgent, '__init__', mock_init)
    mocker.patch('conformer_rl.agents.base_ac_agent.BaseACAgent._sample')
    mocker.patch('conformer_rl.agents.base_ac_agent.BaseACAgent._calculate_advantages')
    mocker.patch('conformer_rl.agents.base_ac_agent.BaseACAgent._train')

    agent = BaseACAgent()
    agent.storage = storage
    agent.step()

    conformer_rl.agents.base_ac_agent.BaseACAgent._sample.assert_called_once()
    conformer_rl.agents.base_ac_agent.BaseACAgent._calculate_advantages.assert_called_once()
    conformer_rl.agents.base_ac_agent.BaseACAgent._train.assert_called_once()
    storage.reset.assert_called_once()

def test_sample(mocker):
    mocker.patch.object(conformer_rl.agents.base_ac_agent.BaseACAgent, '__init__', mock_init)
    mocker.patch('conformer_rl.agents.base_ac_agent.to_np')

    storage = mocker.Mock()

    network = mocker.Mock(return_value = {'a': 'action'})

    task = mocker.Mock()
    task.step.return_value = ('next_states', [1] * 7, np.array([1, 1, 0, 0, 1, 0, 0]), '.')


    config = mocker.Mock()
    config.rollout_length = 4
    config.num_workers = 7

    train_logger = mocker.Mock()

    agent = BaseACAgent()
    agent.storage = storage
    agent.config = config
    agent.total_steps = 0
    agent.states = 'states'
    agent.task = task
    agent.total_rewards = np.array([0] * 7)
    agent.train_logger = train_logger
    agent.network = network

    agent._sample()

    assert(agent.total_steps == 28)
    assert(np.array_equal(agent.total_rewards, np.array([0, 0, 4, 4, 0, 4, 4])))
    assert(train_logger.add_scalar.call_count == 12)

def test_calculate_advantages_sarsa(mocker):
    mocker.patch.object(conformer_rl.agents.base_ac_agent.BaseACAgent, '__init__', mock_init)

    config = mocker.Mock()
    config.num_workers = 1
    config.rollout_length = 7
    config.use_gae = False
    config.discount = 0.75
    

    storage = {}
    storage['r'] = torch.tensor([12, 14, 29, 15, 10, 5, 19, 29])
    storage['m'] = torch.tensor([1, 1, 0, 1, 0, 0, 1, 0])
    storage['v'] = torch.tensor([4, 8, 1, 2, 4, 3, 7, 2])



    agent = BaseACAgent()
    agent.storage = storage
    agent.config = config
    agent.prediction = {'v': torch.tensor([2])}
    agent._calculate_advantages()

    assert(torch.sum(torch.abs(torch.tensor([34.8125, 27.75, 28, 20.5, 6, 2, 13.5]).unsqueeze(0) - torch.tensor(agent.advantages))) < 1e-5)
    assert(torch.sum(torch.abs(torch.tensor([38.8125, 35.75, 29, 22.5, 10, 5, 20.5]).unsqueeze(0) - torch.tensor(agent.returns))) < 1e-5)

def test_calculate_advantages_gae(mocker):
    mocker.patch.object(conformer_rl.agents.base_ac_agent.BaseACAgent, '__init__', mock_init)

    config = mocker.Mock()
    config.num_workers = 1
    config.rollout_length = 7
    config.use_gae = True
    config.gae_lambda = 0.5
    config.discount = 0.75
    

    storage = {}
    storage['r'] = torch.tensor([12, 14, 29, 15, 10, 5, 19, 29])
    storage['m'] = torch.tensor([1, 1, 0, 1, 0, 0, 1, 0])
    storage['v'] = torch.tensor([4, 8, 1, 2, 4, 3, 7, 2])

    agent = BaseACAgent()
    agent.storage = storage
    agent.config = config
    agent.prediction = {'v': torch.tensor([2])}
    agent._calculate_advantages()


    assert(torch.sum(torch.abs(torch.tensor([38.8125, 35.75, 29, 22.5, 10, 5, 20.5]).unsqueeze(0) - torch.tensor(agent.returns))) < 1e-5)
    print(agent.advantages)
    assert(torch.sum(torch.abs(torch.tensor([20.46875, 17.25, 28, 18.25, 6, 2, 13.5]).unsqueeze(0) - torch.tensor(agent.advantages))) < 1e-5)

def test_train(mocker):
    mocker.patch.object(conformer_rl.agents.base_ac_agent.BaseACAgent, '__init__', mock_init)
    agent = BaseACAgent()
    with pytest.raises(NotImplementedError):
        agent._train()



