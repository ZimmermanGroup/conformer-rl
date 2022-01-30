import numpy as np
from rdkit import Chem
import stk
import xarray as xr
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMoleculeConfs
from conformer_rl.analysis.lignin_contacts import (CONF_ID, FUNC_GROUP_ID_1,
                                                   setup_dist_matrices)
from stk.molecular.functional_groups.factories.smarts_functional_group_factory import \
    SmartsFunctionalGroupFactory

def init_stk_from_rdkit(rdkit_mol, functional_groups=None, with_Hs=True):
    if with_Hs:
        rdkit_mol = Chem.rdmolops.AddHs(rdkit_mol)
    Chem.rdmolops.Kekulize(rdkit_mol)
    return stk.BuildingBlock.init_from_rdkit_mol(
        rdkit_mol, functional_groups=functional_groups
    )

    
# JOSH - IN PROGRESS
# class FuncGroupDistanceCalculator:
#     def calculate_distances():
#         dist_matrix_2d, dist_matrices_3d = setup_dist_matrices()
#         atom_1_ids = xr.DataArray(
#             [func_group.c_1.get_id() for func_group in functional_groups],
#             dims=FUNC_GROUP_ID_1
#         )
#         dist_matrices_3d.isel(atom_1=atom_1_ids, atom_2=atom_2_ids)

class LigninPericyclicFunctionalGroupFactory(stk.SmartsFunctionalGroupFactory):
    def __init__(self):
        super().__init__('[H]ccOCC[H]', bonders=(1,6), deleters=())

    def get_functional_groups(self, molecule):
        for f in super().get_functional_groups(molecule):
            f_atoms = list(f.get_atoms())
            f.H_phenyl, f.c_1, f.H_alkyl = (f_atoms[i] for i in (0, 1, 6))
            yield f

class LigninPericyclicCalculator:
    factory = LigninPericyclicFunctionalGroupFactory()
    
    def calculate_distances(self, rdkit_mol):
        dist_matrix_2d, dist_matrices_3d = setup_dist_matrices()
        # get the distance between H_alkyl and c_1
        stk_mol = init_stk_from_rdkit(
            rdkit_mol,
            functional_groups=(LigninPericyclicFunctionalGroupFactory(),),
        )
        functional_groups = tuple(stk_mol.get_functional_groups())
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
        # super().__init__('C', bonders=(0,), deleters=())

    def get_functional_groups(self, molecule):
        for f in super().get_functional_groups(molecule):
            f_atoms = list(f.get_atoms())
            f.H, f.O = (f_atoms[i] for i in (0, 3))
            yield f
        

class LigninMaccollCalculator:
    factory = LigninMaccollFunctionalGroupFactory()
    
    def calculate_distances(self, rdkit_mol):
        dist_matrix_2d, dist_matrices_3d = setup_dist_matrices()
        # get the distance between H and O
        stk_mol = init_stk_from_rdkit(
            rdkit_mol,
            functional_groups=(LigninMaccollFunctionalGroupFactory(),),
        )
        functional_groups = tuple(stk_mol.get_functional_groups())
        print(f'{functional_groups = }')

        H_ids = xr.DataArray(
            [func_group.H.get_id() for func_group in functional_groups],
            dims=FUNC_GROUP_ID_1
        )
        O_ids = xr.DataArray(
            [func_group.O.get_id() for func_group in functional_groups],
            dims=FUNC_GROUP_ID_1,
        )
        func_group_distances = xr.Dataset()
        func_group_distances['Lignin Maccoll mechanism distances'] \
            = dist_matrices_3d.isel(atom_1=H_ids, atom_2=O_ids)
            
        # add energies
        # for some strange reason MMFFOptimizeMoleculeConfs affects SMARTS matching, so pass a copy
        energies = np.array(MMFFOptimizeMoleculeConfs(Chem.rdchem.Mol(rdkit_mol), maxIters=0))[:,1]
        func_group_distances['Energies'] = xr.DataArray(energies, dims=CONF_ID)
        return func_group_distances
