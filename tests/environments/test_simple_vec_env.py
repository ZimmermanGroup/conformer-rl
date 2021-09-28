from conformer_rl.environments.simple_vec_env import SimpleVecEnv

class DummyEnv:
    def __init__(self):
        self.steps = 0

    def step(self, action):
        done = False
        self.steps += 1
        if self.steps >= 5:
            done = True

        return 'obs', 1, done, {}

    def reset(self):
        self.steps = 0

    def render(self):
        return self.steps

    def close(self):
        pass

def test_env():
    def env_fn():
        return DummyEnv()

    env_fns = [env_fn] * 5
    env = SimpleVecEnv(env_fns)
    assert(env.num_envs == 5)

    actions = [1]*5

    obs, rew, done, info = env.step(actions)

    for i in range(5):
        assert obs[i] == 'obs'
        assert rew[i] == 1
        assert done[i] == False
        assert info[i] == {}

    for i in range(4):
        obs, rew, done, info = env.step(actions)

    for i in range(5):
        assert done[i] == True

    rend = env.render()
    for i in range(5):
        assert rend[i] == 0

    rend = env.env_method('render')
    for i in range(5):
        assert rend[i] == 0
    assert len(rend) == 5

    env.reset()
    rend = env.env_method('render')
    for i in range(5):
        assert rend[i] == 0
    assert len(rend) == 5
    env.close()

    
    


