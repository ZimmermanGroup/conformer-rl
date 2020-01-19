import graphenvironments
import gymspace
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
     id='DiffStupid-v0',
     entry_point='graphenvironments:DifferentCarbonSetStupid',
     max_episode_steps=1000,
)

gym.envs.register(
     id='DiffDense-v0',
     entry_point='graphenvironments:DifferentCarbonSetDense',
     max_episode_steps=1000,
)

gym.envs.register(
     id='TestSet-v0',
     entry_point='graphenvironments:TestSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='InOrderTestSet-v0',
     entry_point='graphenvironments:InOrderTestSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='UnholierSetEnergyScaled-v0',
     entry_point='graphenvironments:UnholierSetEnergyScaled',
     max_episode_steps=1000,
)

gym.envs.register(
     id='TwoSet-v0',
     entry_point='graphenvironments:TwoSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='OneSet-v0',
     entry_point='graphenvironments:OneSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='AnotherOneSet-v0',
     entry_point='graphenvironments:AnotherOneSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='ThreeSet-v0',
     entry_point='graphenvironments:ThreeSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='FourSet-v0',
     entry_point='graphenvironments:FourSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='OneSetUN-v0',
     entry_point='graphenvironments:OneSetUN',
     max_episode_steps=1000,
)

gym.envs.register(
     id='GiantSet-v0',
     entry_point='graphenvironments:GiantSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='AllThreeTorsionSet-v0',
     entry_point='graphenvironments:AllThreeTorsionSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='AllFiveTorsionSet-v0',
     entry_point='graphenvironments:AllFiveTorsionSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='AllEightTorsionSet-v0',
     entry_point='graphenvironments:AllEightTorsionSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='AllEightTorsionSetStupid-v0',
     entry_point='graphenvironments:AllEightTorsionSetStupid',
     max_episode_steps=1000,
)

gym.envs.register(
     id='AllEightTorsionSetDense-v0',
     entry_point='graphenvironments:AllEightTorsionSetDense',
     max_episode_steps=1000,
)

gym.envs.register(
     id='AllTenTorsionSet-v0',
     entry_point='graphenvironments:AllTenTorsionSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='TenTorsionSetCurriculum-v0',
     entry_point='graphenvironments:TenTorsionSetCurriculum',
     max_episode_steps=1000,
)


gym.envs.register(
     id='LigninSpacedEnvironment-v0',
     entry_point='gymspace:LigninSpacedEnvironment',
     max_episode_steps=1000,
)
