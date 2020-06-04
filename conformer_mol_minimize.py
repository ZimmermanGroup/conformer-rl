from utils import *
from rdkit import Chem
import pickle
import os

confgen = ConformerGeneratorCustom(max_conformers=1, 
                 rmsd_threshold=None, 
                 force_field='mmff',
                 pool_multiplier=1)


if __name__ == '__main__':
    with open ('sgld_mol_ecutoff.pkl', 'rb') as fp: 
        m3 = pickle.load(fp)
    with open ('test_mol28.pickle', 'rb') as fp: 
        m0 = pickle.load(fp)
    # with open ('test_mol11.pickle', 'rb') as fp: 
    #     m1 = pickle.load(fp)
    with open ('md_mol_ecutoff.pkl', 'rb') as fp: 
        m2 = pickle.load(fp)


        

    minfo = {"molfile": "8_0.mol", "standard": 148.6097477747475, "total": 1.647274557813102}
    energy_norm = minfo['standard']
    gibbs_norm = minfo['total']

    # energys = confgen.get_conformer_energies(m) * 0.25
    # energys = np.sort(energys)
    # print(energys)
    # total = np.sum(np.exp(-(energys-energy_norm)))
    # total /= gibbs_norm

    mols = [m3, m0, m2]
    totals = []

    standard = None
    total_norm = None

    for mol in mols:
        energies = confgen.get_conformer_energies(mol) 

        if standard is None:
            standard = energies.min()

        energies = (energies - standard) * (1/3.97)
        total = np.sum(np.exp(-energies))

        if total_norm is None:
            total_norm = total

        totals.append(total / total_norm)

    tots = np.array(totals)
    print(tots)
    print(standard, total_norm)