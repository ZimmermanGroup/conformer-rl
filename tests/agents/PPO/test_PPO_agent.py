import conformer_rl
import torch
from conformer_rl.agents.PPO.PPO_agent import PPOAgent

def mock_init(self):
    pass

def test_step(mocker):
    mocker.patch('conformer_rl.agents.PPO.PPO_agent.PPOAgent.__init__', mock_init)
    no_grad = mocker.patch('conformer_rl.agents.PPO.PPO_agent.torch.no_grad')
    sample = mocker.patch('conformer_rl.agents.PPO.PPO_agent.PPOAgent._sample')
    advantages = mocker.patch('conformer_rl.agents.PPO.PPO_agent.PPOAgent._calculate_advantages')
    train = mocker.patch('conformer_rl.agents.PPO.PPO_agent.PPOAgent._train')

    agent = PPOAgent()
    agent.storage = mocker.Mock()
    agent.step()

    agent.storage.reset.assert_called_once()
    no_grad.assert_called_once()
    sample.assert_called_once()
    advantages.assert_called_once()
    train.assert_called_once()


def test_train1(mocker):
    mocker.patch('conformer_rl.agents.PPO.PPO_agent.PPOAgent.__init__', mock_init)
    backward = mocker.patch('torch.Tensor.backward')
    nn = mocker.patch('conformer_rl.agents.PPO.PPO_agent.nn')

    def same(arg):
        return arg

    mocker.patch('conformer_rl.agents.PPO.PPO_agent.np.random.permutation', same)

    config = mocker.Mock()
    config.num_workers = 3
    config.rollout_length = 2
    config.entropy_weight = 11.
    config.value_loss_weight = 113
    config.optimization_epochs = 1
    config.mini_batch_size = 4
    config.ppo_ratio_clip = 0.2

    network = mocker.Mock()
    network.side_effect = [{
        'a': torch.rand((4, 6)),
        'log_pi_a': torch.tensor([[-2., -5, -7, -1, -12, -13]] * 4),
        'v': torch.tensor([0., 0.5, 2, 4]).unsqueeze(0),
        'ent': torch.tensor([6., 4, 9, 3]).unsqueeze(0)
    },
    {
        'a': torch.rand((2, 6)),
        'log_pi_a': torch.tensor([[-2., -5, -7, -1, -12, -13]] * 2),
        'v': torch.tensor([7., 0.75]).unsqueeze(0),
        'ent': torch.tensor([8., 18]).unsqueeze(0)
    }
    ]

    def storage_order(key):
        d = {
            'states': ['state'] * 6,
            'a': torch.rand((2*3, 6)),
            'log_pi_a': torch.tensor([[0., 1, 2, 3, 4, 5]] * 6),
        }
        if key not in d:
            return None
        return d[key]
    storage = mocker.Mock()
    storage.order.side_effect = storage_order
    

    agent = PPOAgent()
    agent.returns = [torch.tensor([3., 6, 4]), torch.tensor([2., 7, 13])]
    agent.advantages = [torch.tensor([11., 5, 17]), torch.tensor([15., 1, 3])]
    agent.optimizer = mocker.Mock()
    agent.train_logger = mocker.Mock()
    agent.total_steps = 0
    agent.storage = storage
    agent.config = config
    agent.network = network
    
    agent._train()
    assert(agent.optimizer.zero_grad.call_count == 2)
    assert(agent.optimizer.step.call_count == 2)
    
    args_list = agent.train_logger.add_scalar.call_args_list

    # losses for the first 4 elements in the batch
    assert(abs(args_list[1][0][1] - 5.5) < 1e-3)
    assert(abs(args_list[2][0][1] - (-60.16629)) < 1e-3)
    assert(abs(args_list[3][0][1] - (7.46875)) < 1e-3)
    assert(abs(args_list[4][0][1].item() - 783.80249) < 1e-3)
    assert(backward.call_count == 2)

def test_train2(mocker):
    mocker.patch('conformer_rl.agents.PPO.PPO_agent.PPOAgent.__init__', mock_init)
    backward = mocker.patch('torch.Tensor.backward')
    nn = mocker.patch('conformer_rl.agents.PPO.PPO_agent.nn')

    def same(arg):
        return arg

    mocker.patch('conformer_rl.agents.PPO.PPO_agent.np.random.permutation', same)

    config = mocker.Mock()
    config.num_workers = 3
    config.rollout_length = 2
    config.entropy_weight = 11.
    config.value_loss_weight = 113
    config.optimization_epochs = 1
    config.mini_batch_size = 6
    config.ppo_ratio_clip = 0.2

    network = mocker.Mock()
    network.return_value = {
        'a': torch.rand((2*3, 6)),
        'log_pi_a': torch.tensor([[-2., -5, -7, -1, -12, -13]] * 6),
        'v': torch.tensor([0., 0.5, 2, 4, 7, 0.75]).unsqueeze(0),
        'ent': torch.tensor([6., 4, 9, 3, 8, 18]).unsqueeze(0)
    }

    def storage_order(key):
        d = {
            'states': ['state'] * 6,
            'a': torch.rand((2*3, 6)),
            'log_pi_a': torch.tensor([[0., 1, 2, 3, 4, 5]] * 6),
        }
        if key not in d:
            return None
        return d[key]
    storage = mocker.Mock()
    storage.order.side_effect = storage_order
    

    agent = PPOAgent()
    agent.returns = [torch.tensor([3., 6, 4]), torch.tensor([2., 7, 13])]
    agent.advantages = [torch.tensor([11., 5, 17]), torch.tensor([15., 1, 3])]
    agent.optimizer = mocker.Mock()
    agent.train_logger = mocker.Mock()
    agent.total_steps = 0
    agent.storage = storage
    agent.config = config
    agent.network = network
    
    agent._train()
    assert(agent.optimizer.zero_grad.call_count == 1)
    assert(agent.optimizer.step.call_count == 1)
    
    args_list = agent.train_logger.add_scalar.call_args_list

    # loss values for batch of all 6 values

    assert(abs(args_list[1][0][1] - 8.) < 1e-3)
    assert(abs(args_list[2][0][1] - (-87.66891)) < 1e-3)
    assert(abs(args_list[3][0][1] - (15.546875)) < 1e-3)
    assert(abs(args_list[4][0][1].item() - 1669.12793) < 1e-3)
    assert(backward.call_count == 1)

