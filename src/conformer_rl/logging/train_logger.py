"""
Train_logger
============
"""
from torch.utils.tensorboard import SummaryWriter
from conformer_rl.utils import mkdir

class TrainLogger:
    """Used by agent for logging agent metrics during training.

    Parameters
    ----------
    tag : str
        Unique tag for identifying the logging session.
    dir : str
        Path to root directory for where logging results should be saved.
    use_tensorboard : bool
        Whether or not to save metrics to Tensorboard.
    use_cache : bool
        Whether or not to keep a running cache of the metrics.
    use_print : bool
        Whether or not to print metrics when logged.
    """
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


    def add_scalar(self, key: str, scalar_value: float, global_step: int=None, walltime: float=None) -> None:
        """Logs a single scalar value.

        Parameters
        ----------
        key : str
            The key associated with the logged value.
        scalar_value : float
            The value to be logged.
        global_step : int
            The current agent step when logging the metric.
        walltime : float
            The current time elapsed since the start of agent training.
        """
        if self.use_tensorboard:
            self.writer.add_scalar(key, scalar_value, global_step, walltime)

        if self.use_cache:
            if key in self.cache:
                self.cache[key][0].append(scalar_value)
                self.cache[key][1].append(global_step)
            else:
                self.cache[key] = [[scalar_value], [global_step]]

        if self.use_print:
            print("step:", global_step, key + ":", scalar_value)



