import pybel as pb
from pathlib import Path
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToMolBlock, MolFromMolFile,\
    MolToSmiles
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMoleculeConfs

class IQmol:
    def __init__(self):
        self.iqmol_scratch_path = Path.home() / 'Desktop' / 'iqmol_scratch'
    
    def update(self):
        self.molecules = {}
        self.optimized_molecules = {}
        for file in self.iqmol_scratch_path.iterdir():
            print(MolToSmiles(MolFromMolFile(str(file))))
            self.molecules[file.stem] = MolFromMolFile(str(file))
            mol = MolFromMolFile(str(file))
            print(MolToMolBlock(mol))
            UFFOptimizeMoleculeConfs(mol)
            print(MolToMolBlock(mol))
            self.optimized_molecules[file.stem] = mol
            print(MolToSmiles(self.optimized_molecules[file.stem]))

if __name__ == '__main__':
    print('Test Conformer ML version 0.0.0')
    IQmol().update()
#     mol = MolFromSmiles('CCC')
#     print(MolToMolBlock(mol))