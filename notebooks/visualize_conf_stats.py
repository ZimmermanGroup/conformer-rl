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
    return df

# generate a dataframe with properties of xor gate conformers from a TorsionNet log directory
path = Path.home() / 'ConformerML' / 'conformer-ml' / 'log' / 'mol_eval_2'/ 'eval5' / 'ep0'
df = pd.DataFrame(dtype=object)
for id in range(50,100):
    mol = Chem.MolFromMolFile(str(path / ('step' + str(id) + '.mol')))
    df = append_mol_df(id, mol, df, num_gates=8)
print(df)

# create altair charts for the values of each torsion individually and the xor configuration
alt.Chart(df).mark_rect().encode(
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

# %%
