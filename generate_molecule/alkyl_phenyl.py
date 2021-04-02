# %%

import math
from itertools import islice
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
    benzene = init_building_block(smiles='FC1=CC=CC=C1', functional_groups=[stk.FluoroFactory()])
    display(Draw.MolToImage(mol_with_atom_index(alkane.to_rdkit_mol()),
                            size=(700, 300)))
    display(Draw.MolToImage(mol_with_atom_index(benzene.to_rdkit_mol()),
                            size=(700, 300)))
    
    alkyl_phenyl = stk.ConstructedMolecule(
        topology_graph=stk.polymer.Linear(
            building_blocks=(alkane, benzene),
            repeating_unit='BAB',
            num_repeating_units=1,
            optimizer=stk.MCHammer(),
        )
    )
    display(Draw.MolToImage(mol_with_atom_index(alkyl_phenyl.to_rdkit_mol()),
                            size=(700, 300)))
    
    build_block = stk.BuildingBlock.init_from_molecule(alkyl_phenyl)
    num_atoms = build_block.get_num_atoms()
    
    functional_groups = []
    def make_functional_group(n):
        return stk.GenericFunctionalGroup(atoms=tuple(stk.C(i) for i in range(n, n + 6)),
                                          bonders=(stk.C(n + 1), stk.C(n + 5)),
                                          deleters=tuple(stk.C(i) for i in range(n + 2, n + 5)))
    
    build_block = build_block.with_functional_groups([make_functional_group(i)
                                                              for i in (0, 6)])
    rot_build_block = build_block.with_rotation_about_axis(math.pi,
                                                           build_block.get_direction(),
                                                           build_block.get_centroid())
    polymer = stk.ConstructedMolecule(
        topology_graph=stk.polymer.Linear(
            building_blocks=(build_block, rot_build_block),
            repeating_unit='AB',
            num_repeating_units=2,
            optimizer=stk.MCHammer(),
        )
    )
    boundary_bonds = [bond_info.get_bond() for bond_info in polymer.get_bond_infos()
                      if bond_info.get_building_block_id() is None]
    for bond in islice(boundary_bonds, 1, len(boundary_bonds), 2):
        bond._order = 2
    
    display(Draw.MolToImage(mol_with_atom_index(polymer.to_rdkit_mol()),
                            size=(700, 300)))
    


# %%
