import envs
from utils import *

if __name__ == '__main__':
    t = AdaTask('TestBestGibbs-v0', num_envs=5, single_process=False)
    t.change_level(True)
