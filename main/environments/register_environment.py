import gym

gym.envs.register(
     id='Diff-v0',
     entry_point='main.environments.environments:Diff',
     max_episode_steps=1000,
)