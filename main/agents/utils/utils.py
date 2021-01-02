import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    return x


def to_np(t):
    return t.cpu().detach().numpy()