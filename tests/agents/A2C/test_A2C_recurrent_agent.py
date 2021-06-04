import conformer_rl
import torch
from conformer_rl.agents.A2C.A2C_recurrent_agent import A2CRecurrentAgent

def mock_init(self):
    pass

def test_init(mocker):
    def inner_init(self, config):
        self.recurrence = 6
    mocker.patch('conformer_rl.agents.A2C.A2C_recurrent_agent.BaseACAgentRecurrent.__init__', inner_init)
    config = mocker.Mock()
    config.rollout_length = 5
    config.num_workers = 12

    agent = A2CRecurrentAgent(config)



def test_train_losses(mocker):
    mocker.patch('conformer_rl.agents.A2C.A2C_recurrent_agent.A2CRecurrentAgent.__init__', mock_init)
    backward = mocker.patch('torch.Tensor.backward')
    nn = mocker.patch('conformer_rl.agents.A2C.A2C_recurrent_agent.nn')

    config = mocker.Mock()
    config.num_workers = 3
    config.rollout_length = 2
    config.entropy_weight = 11.
    config.value_loss_weight = 113

    network = mocker.Mock()
    network.return_value = ({
        'a': torch.rand((2*3, 6)),
        'log_pi_a': torch.tensor([[0., 1, 2, 3, 4, 5]] * 6),
        'v': torch.tensor([0., 0.5, 2, 4, 7, 0.75]).unsqueeze(0),
        'ent': torch.tensor([6., 4, 9, 3, 8, 18]).unsqueeze(0)
    }, None)



    def storage_order(key):
        d = {
            'states': ['state'] * 6,
            'a': torch.rand((2*3, 6)),
        }
        return d[key]
    storage = mocker.Mock()
    storage.order.side_effect = storage_order
    

    agent = A2CRecurrentAgent()
    agent.returns = [torch.tensor([3., 6, 4]), torch.tensor([2., 7, 13])]
    agent.advantages = [torch.tensor([11., 5, 17]), torch.tensor([15., 1, 3])]
    agent.optimizer = mocker.Mock()
    agent.train_logger = mocker.Mock()
    agent.total_steps = 0
    agent.storage = storage
    agent.config = config
    agent.network = network
    agent.num_recurrent_units = 0
    agent.recurrence = 1
    
    agent._train()
    agent.optimizer.zero_grad.assert_called_once()
    agent.optimizer.step.assert_called_once()
    
    args_list = agent.train_logger.add_scalar.call_args_list

    assert(abs(args_list[0][0][1].item() - (-21.666667)) < 1e-5)
    assert(abs(args_list[1][0][1].item() - 15.546875) < 1e-5)
    assert(abs(args_list[2][0][1].item() - 8) < 1e-5)
    assert(abs(args_list[3][0][1].item() - 1647.13025) < 1e-3)
    backward.assert_called_once()

    nn.utils.clip_grad_norm_.assert_called_once()

def test_recurrence(mocker):
    mocker.patch('conformer_rl.agents.A2C.A2C_recurrent_agent.A2CRecurrentAgent.__init__', mock_init)
    backward = mocker.patch('torch.Tensor.backward')
    nn = mocker.patch('conformer_rl.agents.A2C.A2C_recurrent_agent.nn')

    config = mocker.Mock()
    config.num_workers = 3
    config.rollout_length = 2
    config.entropy_weight = 11.
    config.value_loss_weight = 113

    network = mocker.Mock()
    network.return_value = ({
        'a': torch.rand((3, 6)),
        'log_pi_a': torch.tensor([[0., 1, 2, 3, 4, 5]] * 3),
        'v': torch.tensor([0., 0.5, 2]).unsqueeze(0),
        'ent': torch.tensor([6., 4, 9]).unsqueeze(0)
    }, None)

    def storage_order(key):
        d = {
            'states': ['state'] * 6,
            'a': torch.rand((2*3, 6)),
        }
        if key not in d:
            return None
        return d[key]
    storage = mocker.Mock()
    storage.order.side_effect = storage_order
    

    agent = A2CRecurrentAgent()
    agent.returns = [torch.tensor([3., 6, 4]), torch.tensor([2., 7, 13])]
    agent.advantages = [torch.tensor([11., 5, 17]), torch.tensor([15., 1, 3])]
    agent.optimizer = mocker.Mock()
    agent.train_logger = mocker.Mock()
    agent.total_steps = 0
    agent.storage = storage
    agent.config = config
    agent.network = network
    agent.num_recurrent_units = 5
    agent.recurrence = 2
    
    agent._train()
    agent.optimizer.zero_grad.assert_called_once()
    agent.optimizer.step.assert_called_once()

    assert(agent.storage.order.call_count == 7)
    args_list = agent.storage.order.call_args_list
    assert(args_list[0][0][0] == 'a')
    assert(args_list[1][0][0] == 'recurrent_states_0')
    assert(args_list[2][0][0] == 'recurrent_states_1')
    assert(args_list[3][0][0] == 'recurrent_states_2')
    assert(args_list[4][0][0] == 'recurrent_states_3')
    assert(args_list[5][0][0] == 'recurrent_states_4')

    assert(agent.network.call_count == 2)

    backward.assert_called_once()

    nn.utils.clip_grad_norm_.assert_called_once()



