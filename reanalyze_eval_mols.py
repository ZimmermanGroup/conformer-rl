from utils import *
from rdkit import Chem
import pickle
import os

confgen = ConformerGeneratorCustom(max_conformers=1, 
                 rmsd_threshold=None, 
                 force_field='mmff',
                 pool_multiplier=1)

if __name__ == '__main__':
    mols = []

    for i in range(10):
        with open (f'test_mol2{i}.pickle', 'rb') as fp: 
            m = pickle.load(fp)
            mols.append(m)        

    e_0 = 525.8597421636731
    z_0 = 16.154879274306534

    totals = []
    for mol in mols:
        energies = confgen.get_conformer_energies(mol) 

        energies = (energies - e_0) * (1/3.97)
        total = np.sum(np.exp(-energies))

        totals.append(total / z_0)

    tots = np.array(totals)
    print('all', tots)
    print('mean', tots.mean())
    print('std', tots.std())
    print('alllogs', np.log(tots))
    print('meanlog', np.log(tots).mean())
    print('stdlog', np.log(tots).std())
    print(standard, total_norm)
