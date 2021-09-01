# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from functools import partial
import pickle
from pathlib import Path
import numpy as np
from numpy.core.numeric import count_nonzero
from numpy.lib.function_base import disp
from rdkit import Chem
from rdkit.Chem.rdmolops import GetShortestPath, GetDistanceMatrix, Get3DDistanceMatrix
import pandas as pd
import xarray as xr
import altair as alt
from IPython.display import display

# alt.data_transformers.disable_max_rows()
alt.data_transformers.enable('json')

# pickle_path = Path('/export/zimmerman/joshkamm/ConformerML/conformer-ml/src/conformer_rl/analysis/pruned_mol_0.1tfd.pkl')
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

# %%
import hvplot.pandas
display(df.hvplot())
df.hvplot()

# %%
for name in df.columns[2:]:
    chart = alt.Chart(df).mark_rect().encode(
        x='x:O',
        y='y:O',
        color=f'{name}:Q'
    ).configure_view(
        step=4
    )
    display(chart)

# %%

def num_contacts_per_conf(mol, smarts_1, smarts_2, thresh_3d=4, thresh_topological=5):
    FUNC_GROUP_ID_1, ATOM_ID_1 = 'func_group_id_1', 'atom_id_1'
    FUNC_GROUP_ID_2, ATOM_ID_2 = 'func_group_id_2', 'atom_id_2'
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

# %%
alt.data_transformers.enable('default')
for var in [NUM_POTENTIAL_CONTACTS_PER_CONF, CONTACTS_PER_CONF_RATIO]:
    chart = alt.Chart(df).mark_circle(color="#91b6d4").encode(
        x='smarts_1:N',
        y='smarts_2:N',
        size=f'{var}:Q',
    )
    text = alt.Chart(df).mark_text(fontSize=20).encode(
        x='smarts_1:N',
        y='smarts_2:N',
        text=alt.Text(f'{var}:Q', format=',.2r'),
    )
    chart = alt.layer(chart, text).configure_view(
        step=50,
    ).properties(
        width=300,
        height=300,
    )
    display(chart)
# from altair_saver import save
# save(chart, 'test.html')
# alt.renderers.enable('altair_saver', fmts=['vega-lite'])
# chart.save('chart.svg')
# chart

# %%
def func_group_distance(i, j):
    return np.min([dist_matrix_3d[x,y] for x in matches[i] for y in matches[j]])
func_group_dist_matrix = np.zeros((len(matches), len(matches)))
for index, value in np.ndenumerate(func_group_dist_matrix):
    func_group_dist_matrix[index] = func_group_distance(*index)
# func_group_dist_matrix = np.array([[func_group_distance(i,j) for i in range(len(matches))]
#                                    for j in range(len(matches))])
print (func_group_dist_matrix)
min_index = np.unravel_index(np.argmin(func_group_dist_matrix), shape=func_group_dist_matrix.shape)
df = pd.DataFrame.from_records(((matches[index[0]], matches[index[1]], func_group_dist_matrix[index])
                                for (index, x) in np.ndenumerate(func_group_dist_matrix)),
                               columns=['x', 'y', '3d_distance'])
df = df.sort_values(by=['3d_distance'])
display(df)

chart = alt.Chart(df).mark_rect().encode(
    x='x:N',
    y='y:N',
    color=f'3d_distance:Q'
).configure_view(
    step=30
)

display(chart)

# %%

# make a sample distogram
mol_path = 'test.mol'
Chem.rdmolfiles.MolToMolFile(
    mol, str(mol_path))
num_conformers = {data_source: rdkit_mol.GetNumConformers() for data_source, rdkit_mol in rdkit_mols.items()}
df = pd.DataFrame.from_records(((data_source, 1 / num_conformers[data_source],
                                 conf.GetAtomPosition(100).Distance(conf.GetAtomPosition(226)))
                                for data_source in rdkit_mols
                                for conf in rdkit_mols[data_source].GetConformers()),
                               columns=['data_source', 'weight', 'distance'])


# distances = [conf.GetAtomPosition(100).Distance(conf.GetAtomPosition(226))
#              for conf in confs]
alt.Chart(df).mark_bar().encode(
    alt.X('distance:Q', bin=alt.Bin(maxbins=30)),
    y=alt.Y('sum(weight)', axis=alt.Axis(format='%', title=None)),
    row='data_source:N'
).properties(
    width=200,
    height=200
)#.save('chart.png', scale_factor=3.0)
# alt.Chart(df).transform_joinaggregate(
#     total='count(*)'
# ).transform_calculate(
#     pct='1 / datum.total'
# ).mark_bar().encode(
#     alt.X('distance:Q', bin=True),
#     alt.Y('sum(pct):Q', axis=alt.Axis(format='%')),
#     row='data_source:N'
# )

# %%
# looking at Zeke's new molecule
mol = Chem.MolFromMolFile('/export/zimmerman/epunzal/2020DowProj/ligninWithJosh/lignin-kmc/lignin_generation/oligomers/12monomers.mol')
