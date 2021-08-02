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

# pickle_path = Path('/export/zimmerman/joshkamm/ConformerML/conformer-ml/src/conformer_rl/analysis/pruned_mol_0.1tfd.pkl')
pickle_paths = {}
RL, MD = 'RL', 'MD'
pickle_paths[RL] = Path('/export/zimmerman/joshkamm/from_epunzal/neurips/pruned_files_ecutoff500/rl_mol_ecutoff.pkl') # RL conformers
pickle_paths[MD] = Path('/export/zimmerman/joshkamm/from_epunzal/neurips/pruned_files_ecutoff500/sgld_mol_ecutoff.pkl') # MD conformers

rdkit_mols = {}
for data_source, pickle_path in pickle_paths.items():
    with open(pickle_path, 'rb') as pickle_file:
        rdkit_mols[data_source] = pickle.load(pickle_file)

# try getting the atoms from the molecule that are the furtherst apart in the graph
mol = rdkit_mols[RL]
print(f'Distance matrix: {GetDistanceMatrix(mol)}')
print(f'Shortest path: {GetShortestPath(mol, 0, 100)}')
dist_matrix = GetDistanceMatrix(mol)
dist_matrix_3d = Get3DDistanceMatrix(mol)
dist_matrix_ratio = dist_matrix_3d / dist_matrix
df = pd.DataFrame.from_records(((index[0], index[1], dist_matrix[index], dist_matrix_3d[index],
                                 dist_matrix_ratio[index])
                                for (index, x) in np.ndenumerate(dist_matrix)),
                                columns=['x', 'y', 'topological_distance', '3d_distance', 'distance_ratio'])
for name in df.columns[2:]:
    chart = alt.Chart(df).mark_rect().encode(
        x='x:O',
        y='y:O',
        color=f'{name}:Q'
    ).configure_view(
        step=4
    )

    # display(chart)
df = df.sort_values(by=['distance_ratio'])
display(df)


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
alt.data_transformers.disable_max_rows()
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

