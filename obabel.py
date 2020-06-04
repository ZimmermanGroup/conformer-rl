import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing
import logging
import torch
import pandas as pd
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance
import torch_geometric.nn as gnn

from utils import *

import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_rl import *
from deep_rl.component.envs import DummyVecEnv, make_env

import envs

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from concurrent.futures import ProcessPoolExecutor


from models import *
from deep_rl import *
import envs


from rdkit import Chem
import os

import json
from tempfile import TemporaryDirectory
import subprocess
from concurrent.futures import ProcessPoolExecutor

from utils import *
import time

confgen = ConformerGeneratorCustom(max_conformers=1,
                 rmsd_threshold=None,
                 force_field='mmff',
                 pool_multiplier=1)

def run_lignins_obabel(tup):
    smiles, energy_norm, gibbs_norm = tup
    init_dir = os.getcwd()

    with TemporaryDirectory() as td:
        os.chdir(td)

        with open('testing.smi', 'w') as fp:
            fp.write(smiles)

        start = time.time()
        subprocess.check_output('obabel testing.smi -O initial.sdf --gen3d --fast', shell=True)
        subprocess.check_output('obabel initial.sdf -O confs.sdf --confab --conf 1000 --ecutoff 100000000.0 --rcutoff 0.001', shell=True)

        inp = load_from_sdf('confs.sdf')
        mol = inp[0]
        for confmol in inp[1:]:
            c = confmol.GetConformer(id=0)
            mol.AddConformer(c, assignId=True)

        res = AllChem.MMFFOptimizeMoleculeConfs(mol)
        mol = prune_conformers(mol, 0.10)

        energys = (confgen.get_conformer_energies(mol) - energy_norm) * (1/3.97)
        total = np.sum(np.exp(-energys))
        total /= gibbs_norm
        end = time.time()
        os.chdir(init_dir)
        return total, end-start


if __name__ == '__main__':
    outputs = []
    times = []

    m = Chem.MolFromMolFile('lignin_eval_final/8_0.mol', removeHs=False)
    # minfo = {"molfile": "8_0.mol", "standard": 148.6097477747475, "total": 1.647274557813102}

    smi = Chem.MolToSmiles(m)
    standard = 525.8597421636731
    total = 16.154879274306534

    lignin_final_args = (smi, standard, total)

    args_list = [lignin_final_args] * 10
    with ProcessPoolExecutor() as executor:
        out = executor.map(run_lignins_obabel, args_list)

    for a, b in out:
        outputs.append(a)
        times.append(b)

    print('outputs', outputs)
    print('times', times)
