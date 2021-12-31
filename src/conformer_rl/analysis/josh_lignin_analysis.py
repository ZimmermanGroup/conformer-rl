# %%
%reload_ext autoreload
%autoreload 2
from IPython.display import display
import numpy as np
import xarray as xr

from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolToPDBBlock

import stk
from conformer_rl.analysis.lignin_contacts import CONF_ID, FUNC_GROUP_ID_1, setup_mol
from conformer_rl.analysis.lignin_pericyclic import \
    LigninPericyclicCalculator, LigninPericyclicFunctionalGroupFactory, \
    LigninMaccollCalculator, LigninMaccollFunctionalGroupFactory

import hvplot.pandas # noqa
import holoviews as hv
from holoviews import opts
from holoviews.streams import Selection1D
import panel as pn
from panel_chemistry.pane import \
    NGLViewer  # panel_chemistry needs to be imported before you run pn.extension()
from panel_chemistry.pane.ngl_viewer import EXTENSIONS
pn.extension('bokeh', comms='vscode')
pn.extension("ngl_viewer", sizing_mode="stretch_width")

# %%
mol = setup_mol()
mol = Chem.rdmolops.AddHs(mol)
Chem.rdmolops.Kekulize(mol)
stk_mol = stk.BuildingBlock.init_from_rdkit_mol(
    mol,
    functional_groups=(
        LigninMaccollFunctionalGroupFactory(),
    )
)
maccoll_distances = LigninMaccollCalculator().calculate_distances(mol)

def highlighted_mol(func_group_id):
    mol = setup_mol()
    for i, f_group in enumerate(stk_mol.get_functional_groups()):
        atom_H = mol.GetAtomWithIdx(f_group.H.get_id())
        atom_O = mol.GetAtomWithIdx(f_group.O.get_id())
        if i == int(func_group_id):
            atom_H.SetAtomicNum(9)
            atom_O.SetAtomicNum(15)
        else: # gently highlight other functional groups in the same conformer
            atom_H.SetAtomicNum(2)
    return mol

df = maccoll_distances.to_dataframe().reset_index().astype({FUNC_GROUP_ID_1: 'str'})
print(df)
points = df.hvplot.scatter(x='Lignin Maccoll mechanism distances', y='Energies', c='func_group_id_1')
points.opts(
    opts.Scatter(tools=['tap', 'hover'], active_tools=['wheel_zoom'], width=600, height=600,
                 marker='triangle', size=10,)
)

stream = Selection1D(source=points)
@pn.depends(stream.param.index)
def display_mol(index):
    if not index:
        return None
    conf_id = int(df.iloc[index]['conf_id'])
    pdb_block = MolToPDBBlock(highlighted_mol(df.iloc[index][FUNC_GROUP_ID_1]), confId=conf_id)
    viewer = NGLViewer(object=pdb_block, extension='pdb', background="#F7F7F7", min_height=800, sizing_mode="stretch_both")
    return viewer


@pn.depends(stream.param.index)
def index_conf(index):
    return index
app = pn.Row(pn.Column(points, index_conf), display_mol)
app.show()

# %%

# %%
# looking at Zeke's new molecule
mol = Chem.MolFromMolFile('/export/zimmerman/epunzal/2020DowProj/ligninWithJosh/lignin-kmc/lignin_generation/oligomers/12monomers.mol')

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
import rdkit
benzene = rdkit.MolFromSmiles('c1ccccc1')