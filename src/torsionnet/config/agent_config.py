class Config:
    def __init__(self):

        # naming
        self.tag = 'test'

        # training objects
        self.train_env = None
        self.eval_env = None
        self.optimizer_fn = None
        self.network = None
        self.curriculum = None

        # batch hyperparameters
        self.num_workers = 1
        self.rollout_length = None
        self.max_steps = 1000000
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.recurrence = 1

        # training hyperparameters
        self.discount = 0.9999
        self.entropy_weight = 0.001
        self.use_gae = True
        self.gae_tau = 0.95
        self.value_loss_weight = 0.25
        self.gradient_clip = 0.5
        self.ppo_ratio_clip = 0.2

        # logging config
        self.data_dir = 'data'












