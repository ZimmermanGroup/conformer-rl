"""
Generate_xor_gate
=================
"""
from rdkit import Chem
from rdkit.Chem import Draw
import stk
from itertools import cycle, islice

def generate_xor_gate(gate_complexity: int=2, num_gates: int=3) -> Chem.Mol:
    """Generates xorgate molecule.
    """
    xorgate = XorGate(gate_complexity, num_gates)
    xorgate = xorgate.polymer.to_rdkit_mol()
    xorgate.UpdatePropertyCache()
    Chem.rdmolops.FastFindRings(xorgate)
    xorgate = Chem.AddHs(xorgate)
    return xorgate

class XorGate:
    def __init__(self, gate_complexity, num_gates):
        # use stk to construct an XOR gate molecule to design specifications
        self.gate_complexity = gate_complexity
        self.num_gates = num_gates
        self.num_torsions = num_gates * gate_complexity

        # construct XOR gate monomers
        xor_gate_top = self.make_xor_monomer(position=0)
        xor_gate_bottom = self.make_xor_monomer(position=3)
        
        # Example: for gate_complexity == 2, num_gates == 5, gives 'AABBAABBAA'
        monomer_pattern = ''.join(islice(cycle('A' * gate_complexity + 'B' * gate_complexity),
                                 num_gates * gate_complexity))
        self.polymer = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=(xor_gate_top, xor_gate_bottom),
                repeating_unit=monomer_pattern,
                num_repeating_units=1,
            )
        )

    def init_building_block(self, smiles):
        mol = stk.BuildingBlock(smiles=smiles)
        mol = Chem.rdmolops.RemoveHs(mol.to_rdkit_mol(), sanitize=True)
        # convert rdkit aromatic bonds to single and double bonds for portability
        Chem.rdmolops.Kekulize(mol)
        return stk.BuildingBlock.init_from_rdkit_mol(mol)

    def make_xor_monomer(self, position=0):
        # initialize building blocks
        benzene = self.init_building_block(smiles='C1=CC=CC=C1')
        acetaldehyde = self.init_building_block(smiles='CC=O')
        benzene = benzene.with_functional_groups([stk.SingleAtom(stk.C(position))])
        acetaldehyde = acetaldehyde.with_functional_groups(
            [stk.SingleAtom(stk.C(1))])

        # construct xor gate monomer
        xor_gate = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=(benzene, acetaldehyde),
                repeating_unit='AB',
                num_repeating_units=1
            )
        )

        # construct functional groups for xor gate monomer
        # numbering starts at top and proceeds clockwise
        c_0, c_1, c_2, c_3, c_4, c_5 = stk.C(0), stk.C(
            1), stk.C(2), stk.C(3), stk.C(4), stk.C(5)
        functional_groups = [stk.GenericFunctionalGroup(atoms=(c_0, c_1, c_2, c_3, c_4, c_5),
                                                        bonders=(c_0, c_3), deleters=(c_4, c_5)),
                            stk.GenericFunctionalGroup(atoms=(c_0, c_1, c_2, c_3, c_4, c_5),
                                                        bonders=(c_1, c_2), deleters=())]
        return stk.BuildingBlock.init_from_molecule(xor_gate, functional_groups=functional_groups)
