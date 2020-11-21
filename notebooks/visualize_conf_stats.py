# %%
from pathlib import Path

# couldn't figure out how to get vscode to search for modules in a different folder
import sys
sys.path.append(str(Path(sys.path[0]).parent))
from utils.moleculeUtilities import get_torsions_degs

from rdkit import Chem

import altair as alt
import pandas as pd

def append_mol_df(id, mol, df, num_gates):
    'adds info about the xor gates in mol to df'
    old_df = df.copy()
    tors_degs = get_torsions_degs(mol)
        
    def get_gate_series(gate):
        'compiles info about a specific xor gate'

        gate_dict = dict()
        gate_dict['id'] = id
        gate_dict['gate'] = gate
        gate_dict['tor_1'] = tors_degs[2*gate]
        gate_dict['tor_2'] = tors_degs[2*gate+1]

        # whether each torsion within the xor gate is positive
        gate_dict['tor_1_bool'] = tors_degs[2*gate] >= 0
        gate_dict['tor_2_bool'] = tors_degs[2*gate+1] >= 0

        # are the torsions within the xor gate the same?
        # note: RDKit measures the torsions such that if both torsions have the same sign the
        # oxygens are actually across from each other (which is a lower energy state)
        gate_dict['xor'] = gate_dict['tor_1_bool'] == gate_dict['tor_2_bool']
        return(gate_dict)

    for gate in range(num_gates):
        df = df.append(get_gate_series(gate), ignore_index=True)
        if not get_gate_series(gate)['xor']:
            return old_df
    return df

# generate a dataframe with properties of xor gate conformers from a TorsionNet log directory
# path = Path.home() / 'ConformerML' / 'conformer-ml' / 'log' / 'mol_eval_2'/ 'eval5' / 'ep0'
path = Path.home() / 'ConformerML' / 'conformer-ml' / 'log' / 'xor_gates'/ 'eval25' / 'ep0'
df = pd.DataFrame(dtype=object)
for id in range(0,200):
    mol = Chem.MolFromMolFile(str(path / ('step' + str(id) + '.mol')))
    df = append_mol_df(id, mol, df, num_gates=8)
display(df)

# %%

# create altair charts for the values of each torsion individually and the xor configuration
display(alt.Chart(df).mark_rect().encode(
    y='id:O',
    x='gate:O',
    color='tor_1:Q'
) | alt.Chart(df).mark_rect().encode(
    y='id:O',
    x='gate:O',
    color='tor_2:Q'
) | alt.Chart(df).mark_rect().encode(
    y='id:O',
    x='gate:O',
    color='xor:N'
)
)
def correlation_chart(df):
    return alt.Chart(df).mark_square().encode(
        facet='gate:O',
        x='tor_1_bool:O',
        y='tor_2_bool:O',
        size=alt.Size('count()', scale=alt.Scale(range=[0, 1500]))
    ).properties(
        width=100,
        height=100
    )
display(correlation_chart(df))

# %%
df_temp = df.pivot(index='id', columns='gate', values=['tor_1_bool','tor_2_bool'])
# display(df_temp)
value_counts = df_temp.value_counts().reset_index()
display(value_counts)
df_temp = df_temp.swaplevel(0, 1, axis=1)
# df_temp.columns = df_temp.columns.to_flat_index()
display(df_temp)

# %%
import altair as alt
from vega_datasets import data

# df_iris = data.iris()
df_iris = df_temp
df_iris.columns = df_iris.columns.to_flat_index()
display(df_iris.corr())
corrMatrix = df_iris.corr().reset_index().melt('index')
corrMatrix.columns = ['var1', 'var2', 'correlation']

base = alt.Chart(corrMatrix).transform_filter(
    alt.datum.var1 < alt.datum.var2
).encode(
    x='var1',
    y='var2',
).properties(
    width=alt.Step(30),
    height=alt.Step(30)
)

rects = base.mark_rect().encode(
    color='correlation'
)

text = base.mark_text(
    size=10
).encode(
    text=alt.Text('correlation', format=".2f"),
    color=alt.condition(
        "datum.correlation > 0.5",
        alt.value('white'),
        alt.value('black')
    )
)

rects + text
# %%
