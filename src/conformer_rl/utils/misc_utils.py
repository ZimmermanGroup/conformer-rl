"""
Miscellaneous Utilities
=======================

Miscellaneous utility functions.
"""
import numpy as np
import os
import torch
from pathlib import Path

from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def current_time() -> str:
    """Returns a string containing the current date and time.
    """
    now = datetime.now()
    date_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    return date_string

def set_one_thread() -> None:
    """Sets the number of CPU threads to 1 for each Python process.
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

def mkdir(path: str) -> None:
    """Creates directory specified by the input string.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

def to_np(t: torch.tensor) -> np.array:
    """Converts a PyTorch tensor to a Numpy array.
    """
    return t.cpu().detach().numpy()

def save_model(model: torch.nn.Module, filename: str) -> None:
    """Saves model parameters of a PyTorch neural network to a file.
    """
    state_dict = model.state_dict()
    torch.save(state_dict, filename)

def load_model(model: torch.nn.Module, filename: str) -> None:
    """Loads model parameters of a PyTorch network from a file.
    """
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)
