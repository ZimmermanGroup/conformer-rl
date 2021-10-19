from dataclasses import InitVar, dataclass
from copy import copy

import xarray as xr
import stk
from stk.molecular.functional_groups.factories.utilities import _get_atom_ids

from conformer_rl.analysis.lignin_contacts import FUNC_GROUP_ID_1, setup_dist_matrices

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

class LigninPericyclicFunctionalGroupFactory(stk.FunctionalGroupFactory):
    def get_functional_groups(self, molecule):
        for atom_ids in _get_atom_ids('[H]ccOCC[H]', molecule):
            atoms = tuple(molecule.get_atoms(atom_ids))
            f_group = LigninPericyclicFunctionalGroup(
                *atoms,
                bonders = (atoms[1], atoms[6]),
            )
            yield f_group

class LigninPericyclicCalculator:
    def calculate_distances(self, rdkit_mol):
        dist_matrix_2d, dist_matrices_3d = setup_dist_matrices()
        # get the distance between H_alkyl and c_1
        stk_mol = stk.BuildingBlock.init_from_rdkit_mol(rdkit_mol)
        factory = LigninPericyclicFunctionalGroupFactory()
        functional_groups = tuple(factory.get_functional_groups(stk_mol))
        c_1_ids = xr.DataArray(
            [func_group.c_1.get_id() for func_group in functional_groups],
            dims=FUNC_GROUP_ID_1
        )
        H_alkyl_ids = xr.DataArray(
            [func_group.H_alkyl.get_id() for func_group in functional_groups],
            dims=FUNC_GROUP_ID_1,
        )
        func_group_distances = dist_matrices_3d.isel(atom_1=c_1_ids, atom_2=H_alkyl_ids)
        func_group_distances.name = "Lignin pericyclic mechanism distances"
        # display(func_group_distances)
        return func_group_distances
