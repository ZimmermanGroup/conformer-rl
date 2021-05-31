from rdkit import Chem

from .xor_gate import XorGate
from rdkit.Chem.rdmolops import FastFindRings

class MoleculeWrapper():
    def __init__(self, mol_input, standard, inv_temp=None, total=None, pruning_thresh=0.05, input_type='smiles'):
        self.standard_energy = standard
        self.inv_temp = inv_temp
        self.total = total
        self.pruning_thresh = pruning_thresh
        self.temp_0 = 1

        if input_type == 'smiles':
            self.mol = Chem.MolFromSmiles(mol_input)
            self.mol = Chem.AddHs(self.mol)
        elif input_type == 'file':
            pass
        elif input_type == 'mol':
            self.mol = mol_input
            self.mol.UpdatePropertyCache()
            FastFindRings(self.mol)

        Chem.AllChem.EmbedMultipleConfs(self.mol, numConfs=1)
        Chem.AllChem.MMFFOptimizeMoleculeConfs(self.mol, maxIters=10000)

# Xorgate
xor_gate = XorGate(gate_complexity=2, num_gates=4)
xor_gate = xor_gate.polymer.stk_molecule.to_rdkit_mol()
XORGATE = [MoleculeWrapper(xor_gate, standard=270, input_type='mol')]

