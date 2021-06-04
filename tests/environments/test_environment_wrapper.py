from conformer_rl.environments.environment_wrapper import Task

def test_task_simple(mocker):
    simple_env = mocker.patch('conformer_rl.environments.environment_wrapper.SimpleVecEnv')
    env = Task('CartPole-v0', concurrency=False, num_envs = 5)

    simple_env.assert_called()

def test_task_simple2(mocker):
    make = mocker.patch('conformer_rl.environments.environment_wrapper.gym.make')
    env = Task('CartPole-v0', concurrency=False, num_envs = 5)

    make.assert_called_with('CartPole-v0')

def test_task_parallel(mocker):
    subproc_env = mocker.patch('conformer_rl.environments.environment_wrapper.SubprocVecEnv')
    env = Task('CartPole-v0', concurrency=True, num_envs = 5)

    subproc_env.assert_called()

