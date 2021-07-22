# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pickle
from pathlib import Path
from rdkit import Chem
import pandas as pd
import altair as alt

pickle_path = Path('/export/zimmerman/joshkamm/ConformerML/conformer-ml/src/conformer_rl/analysis/pruned_mol_0.1tfd.pkl')
with open(pickle_path, 'rb') as pickle_file:
    md_mol = pickle.load(pickle_file)
mol_path = pickle_path.parent / 'test.mol'
Chem.rdmolfiles.MolToMolFile(
    md_mol, str(mol_path), confId=6747)
num_conformers = md_mol.GetNumConformers()
confs = list(md_mol.GetConformers())
distances = [conf.GetAtomPosition(100).Distance(conf.GetAtomPosition(226))
             for conf in confs]
df = pd.DataFrame({'distances' : distances})
alt.Chart(df).mark_bar().encode(
    alt.X('distances:Q', bin=True),
    y='count()'
)

