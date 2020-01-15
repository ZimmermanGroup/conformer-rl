import graphenvironments
import gym


gym.envs.register(
     id='Lignin-v0',
     entry_point='graphenvironments:LigninEnv',
     max_episode_steps=1000,
)

gym.envs.register(
     id='Lignin-v1',
     entry_point='graphenvironments:LigninSetEnv',
     max_episode_steps=1000,
)

gym.envs.register(
     id='Carbon-v0',
     entry_point='graphenvironments:BranchedCarbonSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='UnholierSetEnergy-v0',
     entry_point='graphenvironments:UnholierSetEnergy',
     max_episode_steps=1000,
)

gym.envs.register(
     id='UnholierSetEnergyEval-v0',
     entry_point='graphenvironments:UnholierSetEnergyEval',
     max_episode_steps=1000,
)

gym.envs.register(
     id='Trihexyl-v0',
     entry_point='graphenvironments:TrihexylSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='UnholySetEnergy-v0',
     entry_point='graphenvironments:UnholySetEnergy',
     max_episode_steps=1000,
)

gym.envs.register(
     id='UnholySet-v0',
     entry_point='graphenvironments:UnholySet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='UnholierSet-v0',
     entry_point='graphenvironments:UnholierSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='UnholierPointSetEnergy-v0',
     entry_point='graphenvironments:UnholierPointSetEnergy',
     max_episode_steps=1000,
)

gym.envs.register(
     id='Diff-v0',
     entry_point='graphenvironments:DifferentCarbonSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='TestSet-v0',
     entry_point='graphenvironments:TestSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='UnholierSetEnergyScaled-v0',
     entry_point='graphenvironments:UnholierSetEnergyScaled',
     max_episode_steps=1000,
)
