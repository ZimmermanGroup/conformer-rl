import conformer_rl
import torch
from conformer_rl.agents.A2C.A2C_agent import A2CAgent

def mock_init(self):
    pass

def test_train(mocker):
    mocker.patch('conformer_rl.agents.A2C.A2C_agent.A2CAgent.__init__', mock_init)
    backward = mocker.patch('torch.Tensor.backward')
    nn = mocker.patch('conformer_rl.agents.A2C.A2C_agent.nn')

    config = mocker.Mock()
    config.num_workers = 3
    config.rollout_length = 2
    config.entropy_weight = 11.
    config.value_loss_weight = 113



    def storage_order(key):
        d = {
            'a': torch.rand((2*3, 6)),
            'log_pi_a': torch.tensor([[0., 1, 2, 3, 4, 5]] * 6),
            'v': torch.tensor([0., 0.5, 2, 4, 7, 0.75]).unsqueeze(0),
            'ent': torch.tensor([6., 4, 9, 3, 8, 18]).unsqueeze(0)
        }
        return d[key]
    storage = mocker.Mock()
    storage.order.side_effect = storage_order
    

    agent = A2CAgent()
    agent.returns = [torch.tensor([3., 6, 4]), torch.tensor([2., 7, 13])]
    agent.advantages = [torch.tensor([11., 5, 17]), torch.tensor([15., 1, 3])]
    agent.optimizer = mocker.Mock()
    agent.train_logger = mocker.Mock()
    agent.total_steps = 0
    agent.storage = storage
    agent.config = config
    agent.network = mocker.Mock()
    
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



