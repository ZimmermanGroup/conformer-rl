import pybel as pb
from pathlib import Path
import rdkit

class IQmol:
    def __init__(self):
        self.iqmol_scratch_path = Path.home() / 'Desktop' / 'iqmol_scratch'
    
    def update(self):
        self.molecules = {}
        self.optimized_molecules = {}
        for file in self.iqmol_scratch_path.iterdir():
            print(next(pb.readfile('xyz', str(file))))
            self.molecules[file.stem] = next(pb.readfile('xyz', str(file)))
            print(self.molecules[file.stem].write('xyz'))
            self.optimized_molecules[file.stem] = next(pb.readfile('xyz', str(file)))
            self.optimized_molecules[file.stem].localopt()
            print(self.optimized_molecules[file.stem].write('xyz'))

if __name__ == '__main__':
    print('Test Conformer ML version 0.0.0')
    IQmol().update()