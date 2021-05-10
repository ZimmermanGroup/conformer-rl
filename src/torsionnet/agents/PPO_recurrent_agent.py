from collections import deque
from rdkit import Chem
import numpy as np
import numpy.random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torsionnet.agents.agent_utils import Storage
from torsionnet.agents.base_agent import BaseAgent
from torsionnet.utils import to_np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPORecurrentAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.task = config.train_env #gym environment wrapper
        self.hidden_size = config.hidden_size

        self.network = config.network
        self.network.to(device)

        self.optimizer = config.optimizer_fn(self.network.parameters()) #optimization function
        self.batch_num = 0

        self.states = self.task.reset()

        self.hp = torch.zeros(1, self.config.num_workers, self.hidden_size).to(device) #lstm hidden states
        self.cp = torch.zeros(1, self.config.num_workers, self.hidden_size).to(device) #lstm cell states
        self.hv = torch.zeros(1, self.config.num_workers, self.hidden_size).to(device) #lstm hidden states
        self.cv = torch.zeros(1, self.config.num_workers, self.hidden_size).to(device) #lstm cell states
        self.recurrence = self.config.recurrence
        print("running PPO, tag is " + config.tag)

        assert config.rollout_length % self.recurrence == 0
        assert config.mini_batch_size % self.recurrence == 0

        self.curr = config.curriculum
        if self.curr:
            self.reward_buffer = deque([], maxlen=(config.num_workers + self.curr.min_length))

    def eval_step(self, state, done, rstates):
        with torch.no_grad():
            if done:
                prediction, rstates = self.network(state)
            else:
                prediction, rstates = self.network(state, rstates)

            return prediction['a'], rstates

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        done = True
        rstates = None
        ret = None
        current_step = 0
        while ret is None:
            current_step += 1
            action, rstates = self.eval_step(state, done, rstates)
            done = False
            state, reward, done, info = env.step(to_np(action))
            self.eval_logger.log_step(env.render()[0])
            ret = info[0]['episodic_return']
        return ret


    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)

        states_mem = []
        for _ in range(self.config.num_workers):
            states_mem.append([])


        states = self.states

        ##############################################################################################
        #Sampling Loop
        ##############################################################################################
        with torch.no_grad():
            for _ in range(config.rollout_length):

                #add recurrent states (lstm hidden and lstm cell states) to storage
                storage.add({
                    'hp' : self.hp,
                    'cp' : self.cp,
                    'hv' : self.hv,
                    'cv' : self.cv
                })

                #run the neural net once to get prediction
                prediction, (self.hp, self.cp, self.hv, self.cv) = self.network(states, (self.hp, self.cp, self.hv, self.cv))

                #step the environment with the action determined by the prediction
                next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))

                for _, infoDict in enumerate(info):
                    if infoDict['episodic_return'] is not None:
                        print('logging episodic return train...', self.total_steps)
                        self.train_logger.add_scalar('episodic_return_train', infoDict['episodic_return'], self.total_steps)

                #add everything to storage
                storage.add({
                    'a': prediction['a'],
                    'log_pi_a': prediction['log_pi_a'].squeeze(0),
                    'ent': prediction['ent'].squeeze(0),
                    'v': prediction['v'].squeeze(0),
                })
                storage.add({
                    'r': torch.tensor(rewards).unsqueeze(-1).to(device),
                    'm': torch.tensor(1 - terminals).unsqueeze(-1).to(device)
                    })
                for i in range(config.num_workers):
                    states_mem[i].append(states[i])
                states = next_states

                #zero out lstm recurrent state if any of the environments finish
                for i, done in enumerate(terminals):
                    if done:
                        self.hp[0][i] = torch.zeros(self.hidden_size).to(device)
                        self.cp[0][i] = torch.zeros(self.hidden_size).to(device)
                        self.hv[0][i] = torch.zeros(self.hidden_size).to(device)
                        self.cv[0][i] = torch.zeros(self.hidden_size).to(device)

                self.total_steps += config.num_workers

                for ifs in info:
                    if ifs['episodic_return'] is not None and self.curr:
                        self.reward_buffer.append(ifs['episodic_return'])

            if self.curr is not None:
                if len(self.reward_buffer) >= self.curr.min_length + config.num_workers:
                    rewbuf = np.array(self.reward_buffer)[-1 * self.curr.min_length:]
                    conds = rewbuf > self.curr.win_cond

                    if conds.mean() > self.curr.success_percent:
                        self.task.env_method('change_level', True)
                        self.reward_buffer.clear()

                    if conds.mean() < self.curr.fail_percent:
                        self.task.env_method('change_level', False)
                        self.reward_buffer.clear()

            self.states = states

            prediction, _ = self.network(states, (self.hp, self.cp, self.hv, self.cv))

            storage.add({
                'a': prediction['a'],
                'log_pi_a': prediction['log_pi_a'].squeeze(0),
                'ent': prediction['ent'].squeeze(0),
                'v': prediction['v'].squeeze(0),
            })
            storage.placeholder()


        #############################################################################################
        #Calculate advantages and returns and set up for training
        #############################################################################################

        advantages = torch.tensor(np.zeros((config.num_workers, 1))).to(device)
        returns = prediction['v'].squeeze(0).detach()

        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        storage.a = storage.a[:self.config.rollout_length]
        storage.log_pi_a = storage.log_pi_a[:self.config.rollout_length]
        storage.v = storage.v[:self.config.rollout_length]


        actions = torch.stack(storage.a, 1).view(self.config.num_workers * self.config.rollout_length, -1)
        log_probs_old = torch.stack(storage.log_pi_a, 1).view(self.config.num_workers * self.config.rollout_length, -1)
        values = torch.stack(storage.v, 1).view(self.config.num_workers * self.config.rollout_length, -1)
        returns = torch.stack(storage.ret, 1).view(self.config.num_workers * self.config.rollout_length, -1)
        advantages = torch.stack(storage.adv, 1).view(self.config.num_workers * self.config.rollout_length, -1)


        log_probs_old = log_probs_old.detach()
        values = values.detach()

        hp = torch.stack(storage.hp, 2).view(-1, self.hidden_size)
        cp = torch.stack(storage.cp, 2).view(-1, self.hidden_size)
        hv = torch.stack(storage.hv, 2).view(-1, self.hidden_size)
        cv = torch.stack(storage.cv, 2).view(-1, self.hidden_size)


        advantages = (advantages - advantages.mean()) / advantages.std()

        self.train_logger.add_scalar('advantages', advantages.mean(), self.total_steps)

        states = []
        for block in states_mem:
            states.extend(block)



        ############################################################################################
        #Training Loop
        ############################################################################################

        for _ in range(config.optimization_epochs):
            indices = numpy.arange(0, self.config.rollout_length * self.config.num_workers, self.recurrence)
            indices = numpy.random.permutation(indices)

            if self.batch_num % 2 == 1:
                indices = indices[(indices + self.recurrence) % config.rollout_length != 0]
                indices += self.recurrence // 2
            self.batch_num += 1

            num_indices = config.mini_batch_size // self.recurrence
            starting_batch_indices = [indices[i:i+num_indices] for i in range(0, len(indices), num_indices)]
            for starting_indices in starting_batch_indices:
                batch_entropy = 0
                batch_value_loss = 0
                batch_policy_loss = 0
                batch_loss = 0


                sampled_hp = hp[starting_indices].view(1, -1, self.hidden_size)
                sampled_cp = cp[starting_indices].view(1, -1, self.hidden_size)
                sampled_hv = hv[starting_indices].view(1, -1, self.hidden_size)
                sampled_cv = cv[starting_indices].view(1, -1, self.hidden_size)

                for i in range(self.recurrence):
                    sampled_actions = actions[starting_indices + i]
                    sampled_log_probs_old = log_probs_old[starting_indices + i]
                    sampled_values = values[starting_indices + i]
                    sampled_returns = returns[starting_indices + i]
                    sampled_advantages = advantages[starting_indices + i]

                    sampled_states = [states[j] for j in (starting_indices + i)]

                    prediction, (sampled_hp, sampled_cp, sampled_hv, sampled_cv) = self.network(sampled_states, (sampled_hp, sampled_cp, sampled_hv, sampled_cv), sampled_actions)

                    entropy = prediction['ent'].mean()
                    prediction['log_pi_a'] = prediction['log_pi_a'].squeeze(0)
                    prediction['v'] = prediction['v'].squeeze(0)


                    max_action = prediction['log_pi_a'].shape[1]
                    ratio = (prediction['log_pi_a'] - sampled_log_probs_old[:,:max_action]).exp()

                    obj = ratio * sampled_advantages
                    obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                            1.0 + self.config.ppo_ratio_clip) * sampled_advantages

                    policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * entropy

                    value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                    loss = policy_loss + config.value_loss_weight * value_loss

                    batch_entropy += entropy.item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    if i < self.recurrence - 1:
                        for sample_id, batch_id in enumerate(starting_indices):
                            hp[batch_id + i + 1] = sampled_hp[0][sample_id].detach()
                            cp[batch_id + i + 1] = sampled_cp[0][sample_id].detach()
                            hv[batch_id + i + 1] = sampled_hv[0][sample_id].detach()
                            cv[batch_id + i + 1] = sampled_cv[0][sample_id].detach()


                batch_entropy /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                self.train_logger.add_scalar('entropy_loss', batch_entropy, self.total_steps)
                self.train_logger.add_scalar('policy_loss', batch_policy_loss, self.total_steps)
                self.train_logger.add_scalar('value_loss', batch_value_loss, self.total_steps)

                self.optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.optimizer.step()
