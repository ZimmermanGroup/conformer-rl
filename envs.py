import graphenvironments
import gymspace
import gym



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
     id='ThreeSet-v0',
     entry_point='graphenvironments:ThreeSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='ThreeSetSimple-v0',
     entry_point='graphenvironments:ThreeSetSimple',
     max_episode_steps=1000,
)


gym.envs.register(
     id='FourSet-v0',
     entry_point='graphenvironments:FourSet',
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
     id='TenTorsionSetCurriculumBasic-v0',
     entry_point='graphenvironments:TenTorsionSetCurriculumBasic',
     max_episode_steps=1000,
)

gym.envs.register(
     id='LigninSpacedEnvironment-v0',
     entry_point='gymspace:LigninSpacedEnvironment',
     max_episode_steps=1000,
)

gym.envs.register(
     id='LigninSet-v0',
     entry_point='graphenvironments:LigninSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='LigninSmalls-v0',
     entry_point='graphenvironments:LigninSmalls',
     max_episode_steps=1000,
)

gym.envs.register(
     id='LigninFourSet-v0',
     entry_point='graphenvironments:LigninFourSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='LigninThreeSet-v0',
     entry_point='graphenvironments:LigninThreeSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='LigninTwoSet-v0',
     entry_point='graphenvironments:LigninTwoSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='LargeCarbonSet-v0',
     entry_point='graphenvironments:LargeCarbonSet',
     max_episode_steps=1000,
)

gym.envs.register(
     id='Trihexyl-v0',
     entry_point='graphenvironments:Trihexyl',
     max_episode_steps=1000,
)

gym.envs.register(
     id='TrihexylEval-v0',
     entry_point='graphenvironments:TrihexylEval',
     max_episode_steps=1000,
)
