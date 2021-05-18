from generate_molecule.xor_gate import XorGate
import stk
from stko.molecular.torsion.torsion import Torsion

def test_xor_gate():
    xor_gate = XorGate(gate_complexity=2, num_gates=3)
    torsion_list = xor_gate.polymer.get_torsion_list()
    assert torsion_list == [[1, 0, 7, 6],
                            [10, 9, 14, 13],
                            [17, 16, 21, 20],
                            [24, 23, 28, 27],
                            [32, 33, 35, 34],
                            [39, 40, 42, 41]]
    test_torsion_info, *rest = xor_gate.polymer.get_torsion_infos_by_building_block()[2]
    assert str(test_torsion_info.torsion) == str(Torsion(atom1=stk.C(32), atom2=stk.C(33),
                                                         atom3=stk.C(35), atom4=stk.C(34)))
    assert str(test_torsion_info.building_block_torsion) == str(Torsion(atom1=stk.C(2), atom2=stk.C(3),
                                                                        atom3=stk.C(7), atom4=stk.C(6)))
    assert xor_gate.num_torsions == xor_gate.num_gates * xor_gate.gate_complexity
