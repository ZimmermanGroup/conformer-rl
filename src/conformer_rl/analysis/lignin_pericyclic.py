from dataclasses import InitVar, dataclass
from copy import copy
from stk.molecular.functional_groups.factories.smarts_functional_group_factory import SmartsFunctionalGroupFactory

import xarray as xr
import stk
from stk.molecular.functional_groups.factories.utilities import _get_atom_ids

from conformer_rl.analysis.lignin_contacts import FUNC_GROUP_ID_1, func, setup_dist_matrices

@dataclass
class LigninPericyclicFunctionalGroup(stk.GenericFunctionalGroup):
    H_phenyl: stk.Atom
    c_1: stk.Atom
    c_2: stk.Atom
    oxygen: stk.Atom
    C_1: stk.Atom
    C_2: stk.Atom
    H_alkyl: stk.Atom
    
    bonders: InitVar = ()
    deleters: InitVar = ()
        
    def __post_init__(self, bonders, deleters):
        atoms = (self.H_phenyl, self.c_1, self.c_2, self.oxygen, self.C_1, self.C_2, self.H_alkyl)
        super().__init__(
            atoms=atoms,
            bonders=(atoms[1], atoms[6]),
            deleters=deleters,
            placers=bonders
        )

    def clone(self):
        return copy(self)

class LigninPericyclicFunctionalGroupFactory(stk.SmartsFunctionalGroupFactory):
    def __init__(self):
        super().__init__('[H]ccOCC[H]', bonders=(1,6), deleters=())

    def get_functional_groups(self, molecule):
        for f in super().get_functional_groups(molecule):
            f_atoms = list(f.get_atoms())
            f.H_phenyl, f.c_1, f.H_alkyl = (f_atoms[i] for i in (0, 1, 6))
            yield f

class LigninPericyclicCalculator:
    def calculate_distances(self, rdkit_mol):
        dist_matrix_2d, dist_matrices_3d = setup_dist_matrices()
        # get the distance between H_alkyl and c_1
        stk_mol = stk.BuildingBlock.init_from_rdkit_mol(rdkit_mol)
        factory = LigninPericyclicFunctionalGroupFactory()
        functional_groups = tuple(factory.get_functional_groups(stk_mol))
        print(f'{functional_groups = }')

        c_1_ids = xr.DataArray(
            [func_group.c_1.get_id() for func_group in functional_groups],
            dims=FUNC_GROUP_ID_1
        )
        H_alkyl_ids = xr.DataArray(
            [func_group.H_alkyl.get_id() for func_group in functional_groups],
            dims=FUNC_GROUP_ID_1,
        )
        H_phenyl_ids = xr.DataArray(
            [func_group.H_phenyl.get_id() for func_group in functional_groups],
            dims=FUNC_GROUP_ID_1,
        )
        func_group_distances = xr.Dataset()
        func_group_distances['Lignin pericyclic mechanism distances'] \
            = dist_matrices_3d.isel(atom_1=c_1_ids, atom_2=H_alkyl_ids)
        func_group_distances['Lignin pericyclic mechanism inhibition differences'] \
            = (dist_matrices_3d.isel(atom_1=H_alkyl_ids, atom_2=H_phenyl_ids)
               - func_group_distances['Lignin pericyclic mechanism distances'])
        return func_group_distances

class LigninMaccollFunctionalGroupFactory(stk.SmartsFunctionalGroupFactory):
    def __init__(self):
        super().__init__('[H]CCOc', bonders=(0,3), deleters=())

    def get_functional_groups(self, molecule):
        for f in super().get_functional_groups(molecule):
            f_atoms = list(f.get_atoms())
            f.H, f.O = (f_atoms[i] for i in (0, 3))
            yield f
        
def test(rdkit_mol):
    dist_matrix_2d, dist_matrices_3d = setup_dist_matrices()
    stk_mol = stk.BuildingBlock.init_from_rdkit_mol(rdkit_mol)
    factory = SmartsFunctionalGroupFactory('[H]ccOCC[H]', bonders=(), deleters=())
    functional_groups = tuple(factory.get_functional_groups(stk_mol))
    print(f'{functional_groups = }')
    c_1_ids = xr.DataArray(
        [func_group.c_1.get_id() for func_group in functional_groups],
        dims=FUNC_GROUP_ID_1
    )
    H_alkyl_ids = xr.DataArray(
        [func_group.H_alkyl.get_id() for func_group in functional_groups],
        dims=FUNC_GROUP_ID_1,
    )
    H_phenyl_ids = xr.DataArray(
        [func_group.H_phenyl.get_id() for func_group in functional_groups],
        dims=FUNC_GROUP_ID_1,
    )
    func_group_distances = xr.Dataset()
    func_group_distances['Lignin pericyclic mechanism distances'] \
        = dist_matrices_3d.isel(atom_1=c_1_ids, atom_2=H_alkyl_ids)
    func_group_distances['Lignin pericyclic mechanism inhibition distances'] \
        = dist_matrices_3d.isel(atom_1=H_alkyl_ids, atom_2=H_phenyl_ids)
    return func_group_distances
