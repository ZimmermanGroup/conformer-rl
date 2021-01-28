class Config:
    def __init__(self):
        self.task_fn = None
        self.optimizer_fn = None
        self.network = None
        self.discount = None
        self.log_level = 0
        self.tag = 'vanilla'
        self.num_workers = 1
        self.gradient_clip = None
        self.entropy_weight = 0
        self.use_gae = False
        self.gae_tau = 1.0
        self.max_steps = 1000000
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.eval_env = None
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.curriculum = None
        self.hidden_size = None
        self.recurrence = 1
