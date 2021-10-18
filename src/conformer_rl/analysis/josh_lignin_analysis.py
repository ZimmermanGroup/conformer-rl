# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython.display import display
import numpy as np

import altair as alt
import hvplot.xarray  # noqa - adds hvplot methods to xarray objects
import panel as pn
from panel_chemistry.pane import \
    NGLViewer  # panel_chemistry needs to be imported before you run pn.extension()
from panel_chemistry.pane.ngl_viewer import EXTENSIONS
pn.extension('bokeh', comms='vscode')
pn.extension("ngl_viewer", sizing_mode="stretch_width")

import holoviews as hv
from holoviews.streams import Selection1D
hv.extension('bokeh', comms='vscode')

# alt.data_transformers.disable_max_rows()
alt.data_transformers.enable('json')

from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolToPDBBlock
import stk
from conformer_rl.analysis.lignin_pericyclic import \
    LigninPericyclicCalculator, LigninPericyclicFunctionalGroupFactory

mol = Chem.rdmolops.AddHs(mol)
Chem.rdmolops.Kekulize(mol)
stk_mol = stk.BuildingBlock.init_from_rdkit_mol(
    mol,
    functional_groups=(
        LigninPericyclicFunctionalGroupFactory(),
    )
)
print(*stk_mol.get_functional_groups(), sep='\n')

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
pericyclic_distances = LigninPericyclicCalculator().calculate_distances(mol)
pericyclic_distances_numpy = np.array(pericyclic_distances).flatten()
energies = np.random.uniform(size=pericyclic_distances_numpy.size)
display((pericyclic_distances_numpy, energies))
points = hv.Points((pericyclic_distances_numpy, energies))
points.opts(
    tools=['tap', 'hover'], width=600, 
    marker='triangle', size=10, framewise=True,
)
stream = Selection1D(source=points)
@pn.depends(stream.param.index)
def display_mol(index):
    if not index:
        return None
    print(index)
    conf_id = pericyclic_distances.coords['conf_id'][index[0]].item()
    print(conf_id)
    pdb_block = MolToPDBBlock(mol, confId=conf_id)
    viewer = NGLViewer(object=pdb_block, extension='pdb', background="#F7F7F7", min_height=400, sizing_mode="stretch_both")
    return viewer
app = pn.Row(points, display_mol)
# display(histogram)
# hvplot.show(points)
