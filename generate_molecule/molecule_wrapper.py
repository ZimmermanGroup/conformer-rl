from rdkit import Chem

from .xor_gate import XorGate
from rdkit.Chem.rdmolops import FastFindRings

class MoleculeWrapper():
    def __init__(self, mol_input, standard, inv_temp=None, total=None, input_type='smiles'):
        self.standard = standard
        self.inv_temp = inv_temp
        self.total = total

        if input_type == 'smiles':
            self.molecule = Chem.MolFromSmiles(mol_input)
            self.molecule = Chem.AddHs(self.molecule)
        elif input_type == 'file':
            pass
        elif input_type == 'mol':
            self.molecule = mol_input
            self.molecule.UpdatePropertyCache()
            FastFindRings(self.molecule)

            
        res = Chem.AllChem.EmbedMultipleConfs(self.molecule, numConfs=1)
        res = Chem.AllChem.MMFFOptimizeMoleculeConfs(self.molecule)
            
# Diff
DIFF = [MoleculeWrapper(mol_input="CC(CCC)CCCC(CCCC)CC", standard=7.668625034772399, total=13.263723987526067)]

# Xorgate
xor_gate = XorGate(gate_complexity=2, num_gates=3)
xor_gate = xor_gate.polymer.to_rdkit_mol()
XORGATE = [MoleculeWrapper(xor_gate, standard=1., input_type='mol')]

