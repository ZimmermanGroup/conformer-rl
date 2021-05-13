# %%

from typing import Tuple
from rdkit import Chem
from rdkit.Chem import Draw
import stk
from itertools import cycle, islice
from IPython.display import display
from stko.molecular.molecules.constructed_molecule_torsioned import ConstructedMoleculeTorsioned
from stko.molecular.torsions.torsion import Torsion

class XorGate:
    """class to generate an artificial polymer with low energy conformers of an xor pattern
    
    >>> xor_gate = XorGate(gate_complexity=2, num_gates=3)
    >>> xor_gate.polymer.get_torsion_list()
    [[1, 0, 7, 6],
    [10, 9, 14, 13],
    [17, 16, 21, 20],
    [24, 23, 28, 27],
    [32, 33, 35, 34],
    [39, 40, 42, 41]]
    >>> display(Draw.MolToImage(mol_with_atom_index(xor_gate.polymer.stk_molecule.to_rdkit_mol()),
    ...     size=(700,300)))
    <PIL.PngImagePlugin.PngImageFile ...
    
    Accessing the underlying CostructedMoleculeTorsioned allows for mapping from a torsion in
    an xor gate to a corresponding torsion in the individual gate which contains it.
    >>> test_torsion_info, *rest = xor_gate.polymer.get_torsion_infos_by_building_block()[2]
    >>> test_torsion_info.torsion
    Torsion(atom1=C(32), atom2=C(33), atom3=C(35), atom4=C(34))
    >>> test_torsion_info.building_block_torsion
    Torsion(atom1=C(2), atom2=C(3), atom3=C(7), atom4=C(6))
    >>> display(Draw.MolToImage(mol_with_atom_index(
    ...    test_torsion_info.building_block.to_rdkit_mol()), size=(700, 300)))
    <PIL.PngImagePlugin.PngImageFile ...

    >>> xor_gate.num_torsions == xor_gate.num_gates * xor_gate.gate_complexity
    True
    """
    def __init__(self, gate_complexity, num_gates):
        # use stk to construct an XOR gate molecule to design specifications
        self.gate_complexity = gate_complexity
        self.num_gates = num_gates
        self.num_torsions = num_gates * gate_complexity

        # construct XOR gate monomers
        xor_gate_top, top_building_block = self.make_xor_individual_gate(*self.make_xor_monomer(position=0))
        xor_gate_bottom, bottom_building_block = self.make_xor_individual_gate(*self.make_xor_monomer(position=3))
        
        # Example: for num_gates == 5, gives 'ABABA'
        monomer_pattern = ''.join(islice(cycle('A' + 'B'), num_gates))
        if monomer_pattern == 'A':
            self.polymer = xor_gate_top
        else:
            self.polymer = stk.ConstructedMolecule(
                topology_graph=stk.polymer.Linear(
                    building_blocks=(top_building_block, bottom_building_block),
                    repeating_unit=monomer_pattern,
                    num_repeating_units=1,
                )
            )
            self.polymer = ConstructedMoleculeTorsioned(self.polymer)
            self.polymer.transfer_torsions({top_building_block : xor_gate_top,
                                            bottom_building_block : xor_gate_bottom})

    def make_xor_individual_gate(self, xor_monomer, xor_building_block
                                 ) -> Tuple[ConstructedMoleculeTorsioned, stk.BuildingBlock]:
        """create a single section of polymer with all functional groups interacting
        
        Parameters
        ----------
        xor_monomer
            
        """
        individual_gate = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=(xor_building_block,),
                repeating_unit='A',
                num_repeating_units=self.gate_complexity
            )
        )
        
        individual_gate = ConstructedMoleculeTorsioned(individual_gate)
        individual_gate.transfer_torsions({xor_building_block: xor_monomer})
            
        # construct the functional groups of the gate from the functional groups of the monomers
        functional_groups = list(xor_building_block.get_functional_groups())
        atom_maps = individual_gate.atom_maps
        functional_groups = [functional_groups[0].with_atoms(atom_maps[0]),
                             functional_groups[1].with_atoms(atom_maps[self.gate_complexity - 1])]
        
        gate_building_block = stk.BuildingBlock.init_from_molecule(individual_gate.stk_molecule,
                                                                   functional_groups)
        return individual_gate, gate_building_block

    def make_xor_monomer(self, position=0):
        # initialize building blocks
        benzene = init_building_block(smiles='C1=CC=CC=C1')
        acetaldehyde = init_building_block(smiles='CC=O')
        benzene = benzene.with_functional_groups([stk.SingleAtom(stk.C(position))])
        acetaldehyde = acetaldehyde.with_functional_groups(
            [stk.SingleAtom(stk.C(1))])

        # construct xor gate monomer
        xor_monomer = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=(benzene, acetaldehyde),
                repeating_unit='AB',
                num_repeating_units=1
            )
        )
        xor_monomer = ConstructedMoleculeTorsioned(xor_monomer)
        
        # set the initial torsions
        if position == 0:
            xor_monomer.set_torsions([Torsion(stk.C(1), stk.C(0), stk.C(7), stk.C(6))])
        elif position == 3:
            xor_monomer.set_torsions([Torsion(stk.C(2), stk.C(3), stk.C(7), stk.C(6))])

        # construct functional groups for xor gate monomer
        # numbering starts at top and proceeds clockwise
        c_0, c_1, c_2, c_3, c_4, c_5 = stk.C(0), stk.C(1), stk.C(2), stk.C(3), stk.C(4), stk.C(5)
        functional_groups = [stk.GenericFunctionalGroup(atoms=(c_0, c_3, c_4, c_5),
                                                        bonders=(c_0, c_3), deleters=(c_4, c_5)),
                            stk.GenericFunctionalGroup(atoms=(c_1, c_2),
                                                        bonders=(c_1, c_2), deleters=())]
        xor_building_block = stk.BuildingBlock.init_from_molecule(xor_monomer.stk_molecule,
                                                                  functional_groups)
        return xor_monomer, xor_building_block

def mol_with_atom_index(mol):
    """call on an rdkit molecule before visualizing to show atom indices in visualization"""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def init_building_block(smiles, functional_groups=()):
    """construct a building block with hydrogens removed"""
    mol = stk.BuildingBlock(smiles=smiles)
    mol = Chem.rdmolops.RemoveHs(mol.to_rdkit_mol(), sanitize=True)
    # convert rdkit aromatic bonds to single and double bonds for portability
    Chem.rdmolops.Kekulize(mol)
    return stk.BuildingBlock.init_from_rdkit_mol(mol, functional_groups)

if __name__ == "__main__":
    """utilize the doctest module to check tests built into the documentation"""
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS, verbose=True)
    
# %%
