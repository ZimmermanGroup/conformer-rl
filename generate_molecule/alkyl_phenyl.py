# %%

from generate_molecule.xor_gate import init_building_block, mol_with_atom_index
from rdkit import Chem
from rdkit.Chem import Draw
import stk
from IPython.display import display
from stko.molecular.molecules.constructed_molecule_torsioned import ConstructedMoleculeTorsioned
from stko.molecular.torsions.torsion import Torsion


class AlkylPhenylPolymer:
    """a polymer of alkyl chains bridged with phenyl groups"""
    
    def __init__(self):
        pass

if __name__ == '__main__':
    length = 3
    alkane = init_building_block(smiles='F' + 'C' * length + 'F',
                                 functional_groups=[stk.FluoroFactory()])
    benzene = init_building_block(smiles='FC1=CC=CC=C1')
    display(Draw.MolToImage(mol_with_atom_index(alkane.to_rdkit_mol()),
                            size=(700, 300)))
    display(Draw.MolToImage(mol_with_atom_index(benzene.to_rdkit_mol()),
                            size=(700, 300)))
    
    
    

# %%
