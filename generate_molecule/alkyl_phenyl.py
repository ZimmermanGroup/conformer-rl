# %%

import math
from itertools import islice
from generate_molecule.xor_gate import init_building_block, mol_with_atom_index
from rdkit import Chem
from rdkit.Chem import Draw
import nglview
import stk
from IPython.display import display
from stko.molecular.molecules.constructed_molecule_torsioned import ConstructedMoleculeTorsioned
from stko.molecular.torsions.torsion import Torsion


class AlkylPhenylPolymer:
    """a polymer of alkyl chains bridged with phenyl groups
    
    This is a preliminary implementation with a fixed pattern of building blocks due to simplicity
    of cheminformatic manipulation.
    
    >>> ap_polymer = AlkylPhenylPolymer(num_repeating_units=2)
    
    test on the first torsion in the second building block
    >>> test_torsion_info = ap_polymer.polymer.get_torsion_infos_by_building_block()[1][0]
    >>> test_torsion_info.torsion
    Torsion(atom1=C(16), atom2=C(22), atom3=C(23), atom4=C(24))
    >>> test_torsion_info.building_block_torsion
    Torsion(atom1=C(0), atom2=C(12), atom3=C(13), atom4=C(14))
    
    >>> mol = ap_polymer.polymer.stk_molecule.to_rdkit_mol()
    >>> display(Draw.MolToImage(mol_with_atom_index(mol), size=(700, 300)))
    """
    
    def __init__(self, num_repeating_units=2):
        lengths = [7, 12]
        repeating_unit = 'AB'
        
        def make_functional_group(n):
            return stk.GenericFunctionalGroup(atoms=tuple(stk.C(i) for i in range(n, n + 6)),
                                            bonders=(stk.C(n + 1), stk.C(n + 5)),
                                            deleters=tuple(stk.C(i) for i in range(n + 2, n + 5)))
        
        functional_groups = [make_functional_group(i) for i in (0, 6)]
        build_blocks = [self.make_monomer(length).with_functional_groups(functional_groups)
                        for length in lengths]
        build_blocks[1] = build_blocks[1].with_rotation_about_axis(math.pi,
                                                            build_blocks[1].get_direction(),
                                                            build_blocks[1].get_centroid())

        polymer = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=build_blocks,
                repeating_unit=repeating_unit,
                num_repeating_units=num_repeating_units,
                optimizer=stk.MCHammer(),
            )
        )
        
        # clean up the aromatic double bonds
        boundary_bonds = [bond_info.get_bond() for bond_info in polymer.get_bond_infos()
                        if bond_info.get_building_block_id() is None]
        for bond in islice(boundary_bonds, 1, len(boundary_bonds), 2):
            bond._order = 2
        
        self.polymer = ConstructedMoleculeTorsioned(polymer)
        
    
    @staticmethod
    def make_monomer(length):
        """make an alkyl chain with two phenyl end caps
        
        >>> length = 7
        >>> ap_monomer = AlkylPhenylPolymer.make_monomer(length)
        >>> ap_monomer.get_num_atoms() == 6 * 2 + length
        True
        
        >>> display(Draw.MolToImage(mol_with_atom_index(ap_monomer.to_rdkit_mol()),
        ...                        size=(700, 300)))
        """
        alkane = init_building_block(smiles='F' + 'C' * length + 'F',
                                     functional_groups=[stk.FluoroFactory()])
        benzene = init_building_block(
            smiles='FC1=CC=CC=C1', functional_groups=[stk.FluoroFactory()])

        alkyl_phenyl = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=(alkane, benzene),
                repeating_unit='BAB',
                num_repeating_units=1,
                optimizer=stk.MCHammer(),
            )
        )

        return stk.BuildingBlock.init_from_molecule(alkyl_phenyl)


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS, verbose=True)

# %%
