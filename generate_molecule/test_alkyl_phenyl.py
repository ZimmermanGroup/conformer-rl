from generate_molecule.alkyl_phenyl import AlkylPhenylPolymer
import stk
from stko.molecular.torsion.torsion import Torsion


def test_alkyl_phenyl():
    ap_polymer = AlkylPhenylPolymer(num_repeating_units=2)
    test_torsion_info = ap_polymer.polymer.get_torsion_infos_by_building_block()[1][0]
    assert repr(test_torsion_info.torsion) == repr(
        Torsion(atom1=stk.C(16), atom2=stk.C(22), atom3=stk.C(23), atom4=stk.C(24)))
    assert repr(test_torsion_info.building_block_torsion) == repr(
        Torsion(atom1=stk.C(0), atom2=stk.C(12), atom3=stk.C(13), atom4=stk.C(14)))
    
def test_alkyl_phenyl_monomer():
    length = 7
    ap_monomer = AlkylPhenylPolymer.make_monomer(length)
    assert ap_monomer.get_num_atoms() == 6 * 2 + length
