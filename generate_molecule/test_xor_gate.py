from generate_molecule.xor_gate import XorGate

import pytest

def test_xor_gate():
    xor_gate = XorGate(gate_complexity=2, num_gates=3)
    torsion_list = xor_gate.polymer.get_torsion_list()
    assert torsion_list == [[1, 0, 7, 6],
                            [10, 9, 14, 13],
                            [17, 16, 21, 20],
                            [24, 23, 28, 27],
                            [32, 33, 35, 34],
                            [39, 40, 42, 41]]
