from rdkit import Chem
import os

import json
from tempfile import TemporaryDirectory
import subprocess

from utils import *

confgen = ConformerGeneratorCustom(max_conformers=1, 
                 rmsd_threshold=None, 
                 force_field='mmff',
                 pool_multiplier=1)


def save_lignins_obabel(i, smiles):
    print('got here')
    init_dir = os.getcwd()
    print('got here')
    with TemporaryDirectory() as td:
        os.chdir(td)
        
        with open('testing.smi', 'w') as fp:
            fp.write(smiles)
            
        subprocess.check_output('obabel testing.smi -O initial.sdf --gen3d --fast', shell=True)
        subprocess.check_output('obabel initial.sdf -O confs.sdf --confab --conf 1000 --ecutoff 100000000.0 --rcutoff 0.001', shell=True)
    
        inp = load_from_sdf('confs.sdf')
        mol = inp[0]
        for confmol in inp[1:]:
            c = confmol.GetConformer(id=0)
            mol.AddConformer(c, assignId=True)

        res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=-1)
        mol = prune_conformers(mol, 0.05)
        
        energys = confgen.get_conformer_energies(mol) * 0.25
        standard = energys.min()
        total = np.sum(np.exp(-(energys-standard)))
        
        Chem.MolToMolFile(mol, f'{init_dir}/lignin_eval_final/{num_monos_list[i]}_{i}.mol')
    
    os.chdir(init_dir)
    
    out = {
        'molfile': f'{num_monos_list[i]}_{i}.mol',
        'standard': standard,
        'total': total,
    }
    
    with open(f'lignin_eval_final/{num_monos_list[i]}_{i}.json', 'w') as fp:
        json.dump(out, fp)




if __name__ == '__main__':
    lignin_oligomers = []
    m = Chem.MolFromMolFile('lignin_eval_final/zeke_test_lignin8.mol')
    lignin_oligomers.append(m)
    num_monos_list = [8]

    for i, lo in enumerate(lignin_oligomers):
        mol = Chem.AddHs(lo)
        smiles = Chem.MolToSmiles(mol)
        print(i)
        save_lignins_obabel(i, Chem.MolToSmiles(mol))