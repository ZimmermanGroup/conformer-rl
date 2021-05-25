import torch
class Storage:
    def __init__(self, rollout = 0, workers = 0):
        self.storage = {}
        self.rollout = rollout
        self.workers = workers

    def __getitem__(self, key):
        return self.storage[key]

    def append(self, data:dict) -> None:
        for key, val in data.items():
            self.storage.setdefault(key, []).append(val)

    def order(self, key:str):
        if torch.is_tensor(self.storage[key][0]):
            ordered = torch.stack(self.storage[key][:self.rollout], -2)
            shape = ordered.shape
            ordered = ordered.view(*shape[:-3], self.rollout * self.workers, shape[-1])
            return ordered
        else:
            ordered = []
            for i in range(self.workers):
                ordered += [self.storage[key][j][i] for j in range(self.rollout)]
            return ordered


    def reset(self):
        self.storage = {}