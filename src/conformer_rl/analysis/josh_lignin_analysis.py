# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pickle
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetShortestPath, GetDistanceMatrix, Get3DDistanceMatrix
import pandas as pd
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
DISTANCE_2D = 'topological distance'
create_column(DISTANCE_2D, GetDistanceMatrix(mol))
DISTANCE_3D = '3D distance'
create_column(DISTANCE_3D, Get3DDistanceMatrix(mol))
DISTANCE_RATIO = 'distance ratio'
df[DISTANCE_RATIO] = df[DISTANCE_3D] / df[DISTANCE_2D]

# scale up to all conformers
dist_matrices_3d = np.stack([Get3DDistanceMatrix(mol, confId=conf.GetId())
                            for conf in mol.GetConformers()])
AVG_3D_DISTANCE = 'avg 3D distance'
create_column(AVG_3D_DISTANCE, np.mean(dist_matrices_3d, axis=0))
STD_DEV_3D_DISTANCE = 'std dev 3D distance'
create_column(STD_DEV_3D_DISTANCE, np.std(dist_matrices_3d, axis=0))
AVG_3D_DISTANCE_RATIO = 'avg 3D distance ratio'
df[AVG_3D_DISTANCE_RATIO] = df[AVG_3D_DISTANCE] / df[DISTANCE_2D]
STD_DEV_3D_DISTANCE_RATIO = 'std dev 3D distance ratio'
df[STD_DEV_3D_DISTANCE_RATIO] = df[STD_DEV_3D_DISTANCE] / df[DISTANCE_2D]

df.index.names = ['x', 'y']
df.reset_index(inplace=True)
df = df.sort_values(by=[DISTANCE_RATIO])
display(df)

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

# try working with SMARTS
smarts_s = ['[O][H]', '[OD2]([#6])[#6]', '[CX3]=[CX3]']
smarts_mols = [Chem.MolFromSmarts(smarts) for smarts in smarts_s]
print(smarts_mols)
matches = [mol.GetSubstructMatches(smarts_mol) for smarts_mol in smarts_mols]
print(matches)

def num_contacts_per_conf(mol, smarts_1, smarts_2, thresh_3d=4, thresh_topological=5):
    pass


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

