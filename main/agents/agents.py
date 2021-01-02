import time
import torch
from .algorithms.PPO_recurrent_agent import PPORecurrentAgentRecurrence
from .utils.utils import to_np

def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
        if config.log_interval and not agent.total_steps % config.log_interval:
            # agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        # agent.switch_task()

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

class PPORecurrentEvalAgent(PPORecurrentAgentRecurrence):
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
        while True:
            action, rstates = self.eval_step(state, done, rstates)
            done = False
            state, reward, done, info = env.step(to_np(action))
            ret = info[0]['episodic_return']
            if ret is not None:
                break

        return ret
