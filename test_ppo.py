from deep_rl import *
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv
from utils import *

import torch
from torch import nn

class CartPoleACPolicy(nn.Module):
    def __init__(self, hidden_size):
        super(CartPoleACPolicy, self).__init__()
        self.lin0 = nn.Linear(4, hidden_size)
        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.v_layer = nn.Linear(hidden_size, 1)
        self.a_layer = nn.Linear(hidden_size, 2)

    def forward(self, x, action=None):
        out = self.lin0(tensor(x))
        out = self.lin1(out)
        v = self.v_layer(out)
        logits = self.a_layer(out)

        print(logits)
        print(logits.shape)

        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(0)
        entropy = dist.entropy().unsqueeze(0)

        # action = dist.sample().cuda()
        # log_prob = dist.log_prob(action).unsqueeze(0).cuda()
        # entropy = dist.entropy().unsqueeze(0).cuda()

        print(action)
        print(action.shape)
        raise Exception()

        prediction = {
            'a': action,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }

        return prediction

model = CartPoleACPolicy(8)

# A2C
def a2c_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 1

    config.single_process = (config.num_workers == 1)

    # acrobot
    # config.state_dim = 6
    # config.action_dim = 3
    config.state_dim = 4
    config.action_dim = 2

    config.task_fn = lambda: AdaTask(config.game, num_envs=config.num_workers, single_process=config.single_process)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.002)
    config.network_fn = lambda: CategoricalActorCriticNet(
        config.state_dim, config.action_dim, FCBody(config.state_dim, gate=F.tanh))
    # config.network = model
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.state_normalizer = DummyNormalizer()
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 0.5
    run_steps(A2CAgent(config))

# PPO
def ppo_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 20
    config.single_process = (config.num_workers == 1)

    # acrobot
    # config.state_dim = 6
    # config.action_dim = 3

    config.state_dim = 4
    config.action_dim = 2

    config.task_fn = lambda: AdaTask(config.game, num_envs=config.num_workers, single_process=config.single_process)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.002)
    # config.network = model
    config.network_fn = lambda: CategoricalActorCriticNet(
        config.state_dim, config.action_dim, FCBody(config.state_dim, gate=F.tanh))
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.state_normalizer = DummyNormalizer()
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    config.rollout_length = 128
    config.optimization_epochs = 10
    config.mini_batch_size = 32 * config.num_workers
    config.ppo_ratio_clip = 0.2
    config.log_interval = 128 * config.num_workers * 10
    run_steps(PPOAgent(config))

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    select_device(-1)
    # select_device(0)

    game = 'CartPole-v0'
    tag = 'cartpole_a2c_20workers'
    a2c_feature(game=game, tag=tag)
    # ppo_feature(game=game, tag=tag)
