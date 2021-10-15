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
from rdkit.Chem.rdmolfiles import MolToPDBBlock
import pandas as pd
from stk.molecular.molecules import building_block
import xarray as xr
import altair as alt
from IPython.display import display
import hvplot.xarray # noqa - adds hvplot methods to xarray objects
import panel as pn
import panel.widgets as pnw
import nglview as nv
from panel_chemistry.pane import NGLViewer # panel_chemistry needs to be imported before you run pn.extension()
pn.extension('bokeh', comms='vscode')
from panel_chemistry.pane.ngl_viewer import EXTENSIONS
pn.extension("ngl_viewer", sizing_mode="stretch_width")

import holoviews as hv
from holoviews.streams import Selection1D
hv.extension('bokeh', comms='vscode')

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

import stk
import stko
from dataclasses import InitVar, dataclass
from copy import copy, deepcopy
from stk.molecular.functional_groups.factories.utilities import _get_atom_ids
@dataclass
class LigninPericyclicFunctionalGroup(stk.GenericFunctionalGroup):
    H_phenyl: stk.Atom
    c_1: stk.Atom
    c_2: stk.Atom
    oxygen: stk.Atom
    C_1: stk.Atom
    C_2: stk.Atom
    H_alkyl: stk.Atom
    
    bonders: InitVar = ()
    deleters: InitVar = ()
        
    def __post_init__(self, bonders, deleters):
        atoms = (self.H_phenyl, self.c_1, self.c_2, self.oxygen, self.C_1, self.C_2, self.H_alkyl)
        super().__init__(
            atoms=atoms,
            bonders=(atoms[1], atoms[6]),
            deleters=deleters,
            placers=bonders
        )

    def clone(self):
        return copy(self)

class LigninPericyclicFunctionalGroupFactory(stk.FunctionalGroupFactory):
    def get_functional_groups(self, molecule):
        for atom_ids in _get_atom_ids('[H]ccOCC[H]', molecule):
            atoms = tuple(molecule.get_atoms(atom_ids))
            f_group = LigninPericyclicFunctionalGroup(
                *atoms,
                bonders = (atoms[1], atoms[6]),
            )
            yield f_group

class LigninPericyclicCalculator:
    def calculate_distances(self, rdkit_mol):
        # get the distance between H_alkyl and c_1
        stk_mol = stk.BuildingBlock.init_from_rdkit_mol(rdkit_mol)
        factory = LigninPericyclicFunctionalGroupFactory()
        functional_groups = tuple(factory.get_functional_groups(stk_mol))
        c_1_ids = xr.DataArray(
            [func_group.c_1.get_id() for func_group in functional_groups],
            dims=FUNC_GROUP_ID_1
        )
        H_alkyl_ids = xr.DataArray(
            [func_group.H_alkyl.get_id() for func_group in functional_groups],
            dims=FUNC_GROUP_ID_1,
        )
        func_group_distances = dist_matrices_3d.isel(atom_1=c_1_ids, atom_2=H_alkyl_ids)
        func_group_distances.name = "Lignin pericyclic mechanism distances"
        # display(func_group_distances)
        return func_group_distances
        
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
# slider = pnw.IntSlider(name='time', start=0, end=10)
# slider
# print([conformer.GetId() for conformer in mol.GetConformers()])
# display(nv.show_rdkit(mol, conf_id=359))
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
# %%
import numpy as np
import holoviews as hv
from holoviews import opts
from holoviews.streams import Selection1D
from scipy import stats
import panel as pn
hv.extension('bokeh', comms='vscode')


def gen_samples(N, corr=0.8):
    xx = np.array([-0.51, 51.2])
    yy = np.array([0.33, 51.6])
    means = [xx.mean(), yy.mean()]  
    stds = [xx.std() / 3, yy.std() / 3]
    covs = [[stds[0]**2          , stds[0]*stds[1]*corr], 
            [stds[0]*stds[1]*corr,           stds[1]**2]] 

    return np.random.multivariate_normal(means, covs, N)

data = [('Week %d' % (i%10), np.random.rand(), chr(65+np.random.randint(5)), i) for i in range(100)]
sample_data = hv.NdOverlay({i: hv.Points(gen_samples(np.random.randint(1000, 5000), r2))
                            for _, r2, _, i in data})
points = hv.Scatter(data, 'Date', ['r2', 'block', 'id']).redim.range(r2=(0., 1))
stream = Selection1D(source=points)
empty = (hv.Points(np.random.rand(0, 2)) * hv.Slope(0, 0)).relabel('No selection')

def regression(index):
    if not index:
        return empty
    scatter = sample_data[index[0]]
    xs, ys = scatter['x'], scatter['y']
    slope, intercep, rval, pval, std = stats.linregress(xs, ys)
    return (scatter * hv.Slope(slope, intercep)).relabel('r2: %.3f' % slope)

reg = hv.DynamicMap(regression, kdims=[], streams=[stream])

average = hv.Curve(points, 'Date', 'r2').aggregate(function=np.mean)
layout = points * average + reg
layout.opts(
    opts.Curve(color='black'),
    opts.Slope(color='black', framewise=True),
    opts.Scatter(color='block', tools=['tap', 'hover'], width=600, 
                 marker='triangle', cmap='Set1', size=10, framewise=True),
    opts.Points(frame_width=250),
    opts.Overlay(toolbar='above', legend_position='right')
)

# %%
import panel as pn 
from panel_chemistry.pane import NGLViewer # panel_chemistry needs to be imported before you run pn.extension()
from panel_chemistry.pane.ngl_viewer import EXTENSIONS
pn.extension("ngl_viewer", sizing_mode="stretch_width")
# viewer = NGLViewer(object="1CRN", background="#F7F7F7", min_height=700, sizing_mode="stretch_both")
viewer = NGLViewer(object='C:\Users\Joshua\OneDrive\Desktop\iqmol_scratch\test.mol2', background="#F7F7F7", min_height=700, sizing_mode="stretch_both")
settings = pn.Param(
    viewer,
    parameters=["object","extension","representation","color_scheme","custom_color_scheme","effect",],
    name="&#9881;&#65039; Settings"
)
file_input = pn.widgets.FileInput(accept=','.join('.' + s for s in EXTENSIONS[1:]))

def filename_callback(target, event):
    target.extension = event.new.split('.')[1]

def value_callback(target, event):
    target.object = event.new.decode('utf-8')

file_input.link(viewer, callbacks={'value': value_callback, 'filename': filename_callback})

header = pn.widgets.StaticText(value='<b>{0}</b>'.format("&#128190; File Input"))
file_column = pn.layout.Column(header, file_input)


layout = pn.Param(
    viewer,
    parameters=["sizing_mode", "width", "height", "background"],
    name="&#128208; Layout"
)

pn.Row(
    viewer,
    pn.WidgetBox(settings, layout, width=300, sizing_mode="fixed",),
)
pn.template.FastListTemplate(
    site="Panel Chemistry", 
    title="NGLViewer", 
    sidebar=[file_column, settings, layout],
    main=[viewer]
).show()
