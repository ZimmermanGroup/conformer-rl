#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# from ..network import *
from component import *
from BaseAgent import *
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class A2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        if config.network:
            self.network = config.network
        else:
            self.network = config.network_fn()
        self.network.to(device)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):

        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            start = time.time()
            prediction = self.network(config.state_normalizer(states))
            end = time.time()
            print('network time', end-start)

            start = time.time()
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            end = time.time()
            print('step time', end-start)

            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1).to(device),
                         'm': tensor(1 - terminals).unsqueeze(-1).to(device)})

            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(config.state_normalizer(states))
        storage.add(prediction)
        storage.placeholder()

        start = time.time()
        advantages = tensor(np.zeros((config.num_workers, 1))).to(device)
        returns = prediction['v'].detach().to(device)
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                print(advantages.device)
                print(returns.device)
                print(storage.v[i].detach().device)
                advantages = returns - storage.v[i].detach().to(device)
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
        policy_loss = -(log_prob * advantages).mean()
        value_loss = 0.5 * (returns - value).pow(2).mean()
        entropy_loss = entropy.mean()

        end = time.time()
        print('advantages loop time', end-start)

        start = time.time()
        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)


        self.optimizer.step()
        end = time.time()
        print('backwards pass time', end-start)
