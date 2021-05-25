from torch.utils.tensorboard import SummaryWriter
from torsionnet.utils import mkdir

class TrainLogger:
    def __init__(self, tag: str, dir: str = "data", use_tensorboard: bool = True, use_cache: bool = True, use_print: bool = True):
        self.dir = dir
        self.tag = tag
        mkdir(dir)
        self.use_tensorboard = use_tensorboard
        self.use_cache = use_cache
        self.use_print = use_print

        if self.use_tensorboard:
            path = dir + '/' + 'tensorboard_log' + '/' + self.tag + '/'
            mkdir(path)
            self.writer = SummaryWriter(log_dir = path)

        if self.use_cache:
            self.cache = {}


    def add_scalar(self, key: str, scalar_value: float, global_step: int=None, walltime: float=None):
        if self.use_tensorboard:
            self.writer.add_scalar(key, scalar_value, global_step, walltime)

        if self.use_cache:
            if key in self.cache:
                self.cache[key, 0].append(scalar_value)
                self.cache[key, 1].append(global_step)
            else:
                self.cache[key] = [[scalar_value], [global_step]]

        if self.use_print:
            print("step:", global_step, key + ":", scalar_value)



