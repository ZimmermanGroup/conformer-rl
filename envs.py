import graphenvironments
import zipingenvs
import gymspace
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
# gym.envs.register(
#      id='StraightChainTen-v0',
#      entry_point='graphenvironments:StraightChainTen',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='StraightChainTenEval-v0',
#      entry_point='graphenvironments:StraightChainTenEval',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='StraightChainTenEleven-v0',
#      entry_point='graphenvironments:StraightChainTenEleven',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='StraightChainElevenEval-v0',
#      entry_point='graphenvironments:StraightChainElevenEval',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='StraightChainTenElevenTwelve-v0',
#      entry_point='graphenvironments:StraightChainTenElevenTwelve',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='StraightChainTwelveEval-v0',
#      entry_point='graphenvironments:StraightChainTwelveEval',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='Diff-v0',
#      entry_point='graphenvironments:DifferentCarbonSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='Diff11-v0',
#      entry_point='graphenvironments:DifferentCarbonSet11',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='TestSet-v0',
#      entry_point='graphenvironments:TestSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='InOrderTestSet-v0',
#      entry_point='graphenvironments:InOrderTestSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='TwoSet-v0',
#      entry_point='graphenvironments:TwoSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='OneSet-v0',
#      entry_point='graphenvironments:OneSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='ThreeSet-v0',
#      entry_point='graphenvironments:ThreeSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='ThreeSetPruning-v0',
#      entry_point='graphenvironments:ThreeSetPruning',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='FourSet-v0',
#      entry_point='graphenvironments:FourSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='DiffUnique-v0',
#      entry_point='graphenvironments:DifferentCarbonSetUnique',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='DiffEval-v0',
#      entry_point='graphenvironments:DifferentCarbonSkeletonEval',
#      max_episode_steps=1000,
# )


# gym.envs.register(
#      id='FourSetUnique-v0',
#      entry_point='graphenvironments:FourSetUnique',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='OneSetUnique-v0',
#      entry_point='graphenvironments:OneSetUnique',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='OneSetPruning-v0',
#      entry_point='graphenvironments:OneSetPruning',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='GiantSet-v0',
#      entry_point='graphenvironments:GiantSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='AllThreeTorsionSet-v0',
#      entry_point='graphenvironments:AllThreeTorsionSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='AllFiveTorsionSet-v0',
#      entry_point='graphenvironments:AllFiveTorsionSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='AllEightTorsionSet-v0',
#      entry_point='graphenvironments:AllEightTorsionSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='AllEightTorsionSetStupid-v0',
#      entry_point='graphenvironments:AllEightTorsionSetStupid',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='AllEightTorsionSetDense-v0',
#      entry_point='graphenvironments:AllEightTorsionSetDense',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='AllTenTorsionSet-v0',
#      entry_point='graphenvironments:AllTenTorsionSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='AllTenTorsionSetPruning-v0',
#      entry_point='graphenvironments:AllTenTorsionSetPruning',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='TenTorsionSetCurriculum-v0',
#      entry_point='graphenvironments:TenTorsionSetCurriculum',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='TenTorsionSetCurriculumExp-v0',
#      entry_point='graphenvironments:TenTorsionSetCurriculumExp',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='TenTorsionSetCurriculumForgetting-v0',
#      entry_point='graphenvironments:TenTorsionSetCurriculumForgetting',
#      max_episode_steps=1000,
# )



# gym.envs.register(
#      id='TenTorsionSetCurriculumPruning-v0',
#      entry_point='graphenvironments:TenTorsionSetCurriculumPruning',
#      max_episode_steps=1000,
# )

gym.envs.register(
     id='LigninSpacedEnvironment-v0',
     entry_point='gymspace:LigninSpacedEnvironment',
     max_episode_steps=1000,
)

# gym.envs.register(
#      id='LigninTwoSet-v0',
#      entry_point='graphenvironments:LigninTwoSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninThreeSet-v0',
#      entry_point='graphenvironments:LigninThreeSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninTwoSet2-v0',
#      entry_point='graphenvironments:LigninTwoSet2',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninTwoLowTempEval-v0',
#      entry_point='graphenvironments:LigninTwoLowTempEval',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninThreeLowTempEval-v0',
#      entry_point='graphenvironments:LigninThreeLowTempEval',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninTwoSetLowTemp-v0',
#      entry_point='graphenvironments:LigninTwoSetLowTemp',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninThreeSetLowTemp-v0',
#      entry_point='graphenvironments:LigninThreeSetLowTemp',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninFourSetLowTemp-v0',
#      entry_point='graphenvironments:LigninFourSetLowTemp',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninThreeFourHighTempCurriculum-v0',
#      entry_point='graphenvironments:LigninThreeFourHighTempCurriculum',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninThreeFourHighTempSet-v0',
#      entry_point='graphenvironments:LigninThreeFourHighTempSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninFiveSetLowTemp-v0',
#      entry_point='graphenvironments:LigninFiveSetLowTemp',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninAllSet2-v0',
#      entry_point='graphenvironments:LigninAllSet2',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninAllSet-v0',
#      entry_point='graphenvironments:LigninAllSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninAllSetAdaptive-v0',
#      entry_point='graphenvironments:LigninAllSetAdaptive',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LargeCarbonSet-v0',
#      entry_point='graphenvironments:LargeCarbonSet',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='Trihexyl-v0',
#      entry_point='graphenvironments:Trihexyl',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='TrihexylUnique-v0',
#      entry_point='graphenvironments:TrihexylUnique',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='ThreeSetSkeleton-v0',
#      entry_point='graphenvironments:ThreeSetSkeleton',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninSevenSetSkeleton-v0',
#      entry_point='graphenvironments:LigninSevenSetSkeleton',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninEightSetSkeleton-v0',
#      entry_point='graphenvironments:LigninEightSetSkeleton',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='EightLigninEval-v0',
#      entry_point='graphenvironments:EightLigninEval',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='LigninAllSetSkeletonCurriculum-v0',
#      entry_point='graphenvironments:LigninAllSetSkeletonCurriculum',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='NewLigninCurr-v0',
#      entry_point='graphenvironments:NewLigninCurr',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='NewLigninEval-v0',
#      entry_point='graphenvironments:NewLigninEval',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='NewEnergyLigninCurr-v0',
#      entry_point='graphenvironments:NewEnergyLigninCurr',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='NewEnergyLigninEval-v0',
#      entry_point='graphenvironments:NewEnergyLigninEval',
#      max_episode_steps=1000,
# )


# gym.envs.register(
#      id='TestSetCurriculaExtern-v0',
#      entry_point='graphenvironments:TestSetCurriculaExtern',
#      max_episode_steps=1000,
# )

# gym.envs.register(
#      id='TestPruningSetCurriculaExtern-v0',
#      entry_point='graphenvironments:TestPruningSetCurriculaExtern',
#      max_episode_steps=1000,
# )


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


