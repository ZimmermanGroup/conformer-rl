import pickle
from glob import glob
import numpy as np

from utils import *
from concurrent.futures import ProcessPoolExecutor


confgen = ConformerGeneratorCustom(max_conformers=1,
                             rmsd_threshold=None,
                             force_field='mmff',
                             pool_multiplier=1)

data = {"molfile": "8_0.mol", "standard": 148.6097477747475, "total": 1.647274557813102}
temp = 0.25
outputs = [17316538.680567693, 18664818.52303714, 81482642.75631464, 12806649.680186449, 4024235.254089173, 52621982.481011055, 4430915.579388808, 30751951.939407177, 21047497.382528204, 65287117.76117588]
eps = 0.1

def read_func(fn):
    print(fn)

    with open(fn, 'rb') as fp:
        mol = pickle.load(fp)
        energies = confgen.get_conformer_energies(mol)
        score = np.sum(np.exp((energies * 0.25) - data['standard']))/data['total']
        
    for idx, o in enumerate(outputs):
        if np.abs(o - score) < eps:
            return fn, o

    return fn, -1

if __name__ == '__main__':
    fns = list(glob('test_mol*pickle'))
    print(fns)
    with ProcessPoolExecutor() as executor:
        os = executor.map(read_func, fns)

    os = list(os)

    for i, fn in enumerate(fns):
        print(fn, os[i])


        