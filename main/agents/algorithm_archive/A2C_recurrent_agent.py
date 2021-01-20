from component import *
from BaseAgent import *

from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2CRecurrentAgent(BaseAgent):
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
        self.recurrent_states = None
        self.done = True
        self.smh = None

        self.curr = config.curriculum
        if self.curr:
            self.reward_buffer = deque([], maxlen=(config.num_workers + self.curr.min_length))

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            start = time.time()
            if self.done:
                print('done1')
                prediction, self.recurrent_states = self.network(config.state_normalizer(states))
            else:
                prediction, self.recurrent_states = self.network(config.state_normalizer(states), self.recurrent_states)
            end = time.time()

            print('reserved bytes', torch.cuda.memory_reserved() / (1024 * 1024), 'MB')
            self.logger.add_scalar('reserved_bytes', torch.cuda.memory_reserved() / (1024 * 1024), self.total_steps)

            print('forward time', end-start)
            self.logger.add_scalar('forward_pass_time', end-start, self.total_steps)

            self.done = False

            start = time.time()
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            end = time.time()

            self.logger.add_scalar('env_step_time', end-start, self.total_steps)
            print('step time', end-start)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            if terminals[0] == None:
                print(terminals)
                print(rewards)
                print(info)
                print(to_np(prediction['a']))
                print(self.total_steps)
            storage.add({'r': tensor(rewards).unsqueeze(-1).cuda(),
                         'm': tensor(1 - terminals).unsqueeze(-1).cuda()})

            states = next_states

            self.recurrent_states = [rs * storage.m[-1] for rs in self.recurrent_states]
            self.total_steps += config.num_workers

            for ifs in info:
                if ifs['episodic_return'] is not None and self.curr:
                    self.reward_buffer.append(ifs['episodic_return'])

        if self.curr is not None:
            if len(self.reward_buffer) >= self.curr.min_length + config.num_workers:
                rewbuf = np.array(self.reward_buffer)[-1 * self.curr.min_length:]
                conds = rewbuf > self.curr.win_cond

                self.logger.add_scalar('win_percent', conds.mean(), self.total_steps)

                if conds.mean() > self.curr.success_percent:
                    self.task.change_level(True)
                    self.reward_buffer.clear()

                if conds.mean() < self.curr.fail_percent:
                    self.task.change_level(False)
                    self.reward_buffer.clear()

        self.states = states

        with torch.no_grad():
            prediction, _ = self.network(config.state_normalizer(states), self.recurrent_states)

        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1))).cuda()
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        start_train = time.time()

        log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
        policy_loss = -(log_prob * advantages).mean()
        value_loss = 0.5 * (returns - value).pow(2).mean()
        entropy_loss = entropy.mean()

        self.logger.add_scalar('advantages', advantages.mean(), self.total_steps)
        self.logger.add_scalar('policy_loss', policy_loss, self.total_steps)
        self.logger.add_scalar('value_loss', value_loss, self.total_steps)
        self.logger.add_scalar('entropy_loss', entropy_loss, self.total_steps)

        start = time.time()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.logger.add_scalar('grad_norm', grad_norm, self.total_steps)
        self.optimizer.step()

        end = time.time()
        self.logger.add_scalar('backwards_pass_time', end-start, self.total_steps)

        self.recurrent_states = [rs.detach() for rs in self.recurrent_states]

        end_train = time.time()
        self.logger.add_scalar('train_loop_time', end_train-start_train, self.total_steps)



# class A2CEvalAgent(A2CAgent):
#     def eval_step(self, state):
#         prediction = self.network(self.config.state_normalizer(state))
#         return prediction['a']

# class A2CRecurrentEvalAgent(A2CRecurrentAgent):
#     def eval_step(self, state, done, rstates):
#         with torch.no_grad():
#             if done:
#                 prediction, rstates = self.network(self.config.state_normalizer(state))
#             else:
#                 prediction, rstates = self.network(self.config.state_normalizer(state), rstates)

#             return prediction['a'], rstates

#     def eval_episode(self):
#         env = self.config.eval_env
#         state = env.reset()
#         done = True
#         rstates = None
#         while True:
#             action, rstates = self.eval_step(state, done, rstates)
#             done = False
#             state, reward, done, info = env.step(to_np(action))
#             ret = info[0]['episodic_return']
#             if ret is not None:
#                 break

#         return ret