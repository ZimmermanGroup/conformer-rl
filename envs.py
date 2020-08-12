import graphenvironments
import zipingenvs
import gym
import sys
import inspect

clsmembers = inspect.getmembers(sys.modules['graphenvironments'], inspect.isclass)
for i, j in clsmembers:
     if issubclass(j, gym.Env):
          gym.envs.register(
               id=f'{i}-v0',
               entry_point=f'graphenvironments:{i}',
               max_episode_steps=1000,
          )

gym.envs.register(
     id='TestBestGibbs-v0',
     entry_point='zipingenvs:TestBestGibbs',
     max_episode_steps=1000,
)

gym.envs.register(
     id='TChainTrain-v0',
     entry_point='zipingenvs:TChainTrain',
     max_episode_steps=1000,
)


for i in range(0, 10):
     gym.envs.register(
          id=f'TChainTest-v{i}',
          entry_point='zipingenvs:TChainTest',
          max_episode_steps=1000,
          kwargs={'ind_select': i}
     )

for i in range(0, 10):
     gym.envs.register(
          id=f'TChainTest2-v{i}',
          entry_point='zipingenvs:TChainTest2',
          max_episode_steps=1000,
          kwargs={'ind_select': i}
     )

for i in range(0, 10):
     gym.envs.register(
          id=f'TChainTest3-v{i}',
          entry_point='zipingenvs:TChainTest3',
          max_episode_steps=1000,
          kwargs={'ind_select': i}
     )


