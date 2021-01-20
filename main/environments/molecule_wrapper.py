from rdkit import Chem

class MoleculeWrapper():
    def __init__(self, mol_input, standard, inv_temp=None, total=None, input_type='smiles'):
        self.standard = standard
        self.inv_temp = inv_temp
        self.total = total

        if input_type == 'smiles':
            self.molecule = Chem.MolFromSmiles(mol_input)
            self.molecule = Chem.AddHs(self.molecule)
            res = Chem.AllChem.EmbedMultipleConfs(self.molecule, numConfs=1)
            res = Chem.AllChem.MMFFOptimizeMoleculeConfs(self.molecule)
        elif input_type == 'file':
            

DIFF = [MoleculeWrapper(mol_input="CC(CCC)CCCC(CCCC)CC", standard=7.668625034772399, total=13.263723987526067)]
XORGATE = [MoleculeWrapper(mol_input="molecules/xor_gate/XORgateNo8.mol")]

