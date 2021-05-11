from rdkit import Chem

from .xor_gate import XorGate
from rdkit.Chem.rdmolops import FastFindRings

class MoleculeWrapper():
    def __init__(self, mol_input, standard, inv_temp=None, total=None, pruning_thresh=0.05, input_type='smiles'):
        self.standard = standard
        self.inv_temp = inv_temp
        self.total = total
        self.pruning_thresh = pruning_thresh

        if input_type == 'smiles':
            self.molecule = Chem.MolFromSmiles(mol_input)
            self.molecule = Chem.AddHs(self.molecule)
        elif input_type == 'file':
            pass
        elif input_type == 'mol':
            self.molecule = mol_input
            self.molecule.UpdatePropertyCache()
            FastFindRings(self.molecule)

        Chem.AllChem.EmbedMultipleConfs(self.molecule, numConfs=1)
        Chem.AllChem.MMFFOptimizeMoleculeConfs(self.molecule, maxIters=200)
            
# Diff
DIFF = [MoleculeWrapper(mol_input="CC(CCC)CCCC(CCCC)CC", standard=7.668625034772399, total=13.263723987526067)]

# Xorgate
xor_gate = XorGate(gate_complexity=2, num_gates=4)
xor_gate = xor_gate.polymer.stk_molecule.to_rdkit_mol()
XORGATE = [MoleculeWrapper(xor_gate, standard=270, input_type='mol')]

