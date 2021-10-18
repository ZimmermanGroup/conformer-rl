# pickle_path = Path('/export/zimmerman/joshkamm/ConformerML/conformer-ml/src/conformer_rl/analysis/pruned_mol_0.1tfd.pkl')
from functools import partial
from pathlib import Path
import pickle

from IPython.display import display
import numpy as np
import pandas as pd
from rdkit import Chem
import xarray as xr
from rdkit.Chem.rdmolops import (Get3DDistanceMatrix, GetDistanceMatrix,
                                 GetShortestPath)
def setup_dataframe():
    pickle_paths = {}
    RL, MD = 'RL', 'MD'
    pickle_paths[RL] = Path('/export/zimmerman/joshkamm/from_epunzal/neurips/pruned_files_ecutoff500/rl_mol_ecutoff.pkl') # RL conformers
    pickle_paths[MD] = Path('/export/zimmerman/joshkamm/from_epunzal/neurips/pruned_files_ecutoff500/sgld_mol_ecutoff.pkl') # MD conformers

    rdkit_mols = {}
    for data_source, pickle_path in pickle_paths.items():
        with open(pickle_path, 'rb') as pickle_file:
            rdkit_mols[data_source] = pickle.load(pickle_file)

    df = pd.DataFrame()
    def create_column(name, matrix):
        df[name] = pd.DataFrame(matrix).stack(dropna=False)

    mol = rdkit_mols[RL]
    ATOM_1, ATOM_2 = 'atom_1', 'atom_2'
    DISTANCE_2D = 'topological distance'
    dist_matrix_2d = xr.DataArray(GetDistanceMatrix(mol), dims=(ATOM_1, ATOM_2))
    create_column(DISTANCE_2D, GetDistanceMatrix(mol))
    DISTANCE_3D = '3D distance'
    create_column(DISTANCE_3D, Get3DDistanceMatrix(mol))
    DISTANCE_RATIO = 'distance ratio'
    df[DISTANCE_RATIO] = df[DISTANCE_3D] / df[DISTANCE_2D]

    # scale up to all conformers
    CONF_ID = 'conf_id'
    conf_id_index = pd.Index([conf.GetId() for conf in mol.GetConformers()], name=CONF_ID)
    dist_matrices_3d = [xr.DataArray(Get3DDistanceMatrix(mol, confId=conf_id),
                                    dims=(ATOM_1, ATOM_2))
                        for conf_id in conf_id_index]
    dist_matrices_3d = xr.concat(dist_matrices_3d, dim=conf_id_index)
    display(dist_matrices_3d)
    AVG_3D_DISTANCE = 'avg 3D distance'
    df[AVG_3D_DISTANCE] = dist_matrices_3d.mean(dim=CONF_ID).stack(z=(ATOM_1, ATOM_2))
    STD_DEV_3D_DISTANCE = 'std dev 3D distance'
    df[STD_DEV_3D_DISTANCE] = dist_matrices_3d.std(dim=CONF_ID).stack(z=(ATOM_1, ATOM_2))
    AVG_3D_DISTANCE_RATIO = 'avg 3D distance ratio'
    df[AVG_3D_DISTANCE_RATIO] = df[AVG_3D_DISTANCE] / df[DISTANCE_2D]
    STD_DEV_3D_DISTANCE_RATIO = 'std dev 3D distance ratio'
    df[STD_DEV_3D_DISTANCE_RATIO] = df[STD_DEV_3D_DISTANCE] / df[DISTANCE_2D]

    df.index.names = ['x', 'y']
    df.reset_index(inplace=True)
    df = df.sort_values(by=[DISTANCE_RATIO])
    display(df)

FUNC_GROUP_ID_1, ATOM_ID_1 = 'func_group_id_1', 'atom_id_1'
FUNC_GROUP_ID_2, ATOM_ID_2 = 'func_group_id_2', 'atom_id_2'
def num_contacts_per_conf(mol, smarts_1, smarts_2, thresh_3d=4, thresh_topological=5):
    smarts_1_array = xr.DataArray(np.array(matches[smarts_1]), dims=(FUNC_GROUP_ID_1, ATOM_ID_1))
    smarts_2_array = xr.DataArray(np.array(matches[smarts_2]), dims=(FUNC_GROUP_ID_2, ATOM_ID_2))
    all_distances_2d = dist_matrix_2d.isel(atom_1=smarts_1_array, atom_2=smarts_2_array)
    all_distances_3d = dist_matrices_3d.isel(atom_1=smarts_1_array, atom_2=smarts_2_array)
    func_group_distances_2d = all_distances_2d.min(dim=(ATOM_ID_1, ATOM_ID_2))
    func_group_distances_3d = all_distances_3d.min(dim=(ATOM_ID_1, ATOM_ID_2))
    # display(func_group_distances_3d < thresh_3d)
    contacts = (func_group_distances_3d < thresh_3d).where(func_group_distances_2d > thresh_topological,
                                                       other=False)
    return (contacts.reduce(np.count_nonzero, dim=(FUNC_GROUP_ID_1, FUNC_GROUP_ID_2)).mean().item())
            # contacts.count(dim=(FUNC_GROUP_ID_1, FUNC_GROUP_ID_2)).sum().item())
    
def func(arg, **kwargs):
    x, y = arg
    return num_contacts_per_conf(mol, x, y, **kwargs)


# try working with SMARTS
smarts_s = ['[C][OH1]', '[OD2]([c])[c]',
            '[OD2]([CH3])[c]', '[OD2]([CH1])[c]', '[CX3]=[CX3]']
smarts_mols = {smarts: Chem.MolFromSmarts(smarts) for smarts in smarts_s}
matches = {smarts: mol.GetSubstructMatches(smarts_mol)
           for smarts, smarts_mol in smarts_mols.items()}
len_matches = {smarts: len(matches[smarts]) for smarts in smarts_mols}
[print(item) for item in matches.items()]
[print(item) for item in len_matches.items()]
conf_id_index = pd.MultiIndex.from_product([smarts_s, smarts_s], names=['smarts_1', 'smarts_2'])
df = pd.DataFrame(index=conf_id_index)
NUM_CONTACTS_PER_CONF = 'num contacts per conf'
df[NUM_CONTACTS_PER_CONF] = conf_id_index.map(func)
NUM_POTENTIAL_CONTACTS_PER_CONF = 'num potential contacts per conf'
df[NUM_POTENTIAL_CONTACTS_PER_CONF] = conf_id_index.map(partial(func, thresh_3d=1000))
CONTACTS_PER_CONF_RATIO = 'contacts per conf ratio'
df[CONTACTS_PER_CONF_RATIO] = df[NUM_CONTACTS_PER_CONF] / df[NUM_POTENTIAL_CONTACTS_PER_CONF]
df.reset_index(inplace=True)
display(df)
