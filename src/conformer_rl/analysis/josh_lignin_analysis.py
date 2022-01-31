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
    LigninMaccollCalculator, LigninMaccollFunctionalGroupFactory, init_stk_from_rdkit

import param
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

## %%
class LigninDashboard(param.Parameterized):
    # mechanism = param.Selector(['Pericylic', 'Maccoll'])
    mechanism = param.ObjectSelector(default=LigninPericyclicCalculator,
                                     objects=[LigninPericyclicCalculator, LigninMaccollCalculator])
    func_group_factories = {LigninPericyclicCalculator: LigninPericyclicFunctionalGroupFactory,
                            LigninMaccollCalculator: LigninMaccollFunctionalGroupFactory}

    def __init__(self):
        super().__init__()
        self.mol = setup_mol()
        self.dataframe()
        self.stream = Selection1D()
    
    @param.depends('mechanism', watch=True)
    def dataframe(self):
        maccoll_distances = self.mechanism().calculate_distances(self.mol)
        self.df = maccoll_distances.to_dataframe().reset_index().astype({FUNC_GROUP_ID_1: 'str'})
    
    @param.depends('dataframe', 'setup_stk_mol')
    def scatter_plot(self):
        if self.mechanism == LigninMaccollCalculator:
            points = self.df.hvplot.scatter(x='Lignin Maccoll mechanism distances', y='Energies', c='func_group_id_1')
        elif self.mechanism == LigninPericyclicCalculator:
            points = self.df.hvplot.scatter(x='Lignin pericyclic mechanism distances',
                                            y='Lignin pericyclic mechanism inhibition differences')
        points.opts(
            opts.Scatter(tools=['tap', 'hover'], active_tools=['wheel_zoom'], width=600, height=600,
                        marker='triangle', size=10,)
        )
        self.stream.update(index=[])
        self.stream.source = points
        return points
    
    param.depends('display_mol', 'scatter_plot', 'index_conf', 'disp_mechanism', 'mechanism')
    def app(self):
        return pn.Row(pn.Column(self.param.mechanism, self.scatter_plot, self.index_conf, self.disp_mechanism), self.display_mol)
    
    @param.depends('mechanism', watch=True)
    def setup_stk_mol(self):
        return init_stk_from_rdkit(
            self.mol,
            functional_groups=(self.func_group_factories[self.mechanism](),),
        )
        
    @param.depends('stream.index', 'scatter_plot', watch=True)
    def display_mol(self):
        index = self.stream.index
        if not index:
            return None
        index = index[0]
        conf_id = int(self.df.iloc[index]['conf_id'])
        pdb_block = MolToPDBBlock(self.highlighted_mol(self.df.iloc[index][FUNC_GROUP_ID_1]), confId=conf_id)
        viewer = NGLViewer(object=pdb_block, extension='pdb', background="#F7F7F7", min_height=800, sizing_mode="stretch_both")
        return viewer

    @param.depends('stream.index', 'mechanism', watch=True)
    def index_conf(self):
        # index = self.stream.index
        # return index
        return f'{self.stream.index = }\n{repr(dashboard) = }'
    
    @param.depends('mechanism', watch=True)
    def disp_mechanism(self):
        return f'{self.param.mechanism = }'

    def highlighted_mol(self, func_group_id):
        mol = setup_mol()
        for i, f_group in enumerate(self.setup_stk_mol().get_functional_groups()):
            if self.mechanism == LigninMaccollCalculator:
                atom_1 = mol.GetAtomWithIdx(f_group.H.get_id())
                atom_2 = mol.GetAtomWithIdx(f_group.O.get_id())
                if i == int(func_group_id):
                    atom_1.SetAtomicNum(9)
                    atom_2.SetAtomicNum(15)
                else: # gently highlight other functional groups in the same conformer
                    atom_1.SetAtomicNum(2)
            elif self.mechanism == LigninPericyclicCalculator:
                atom_1 = mol.GetAtomWithIdx(f_group.H_phenyl.get_id())
                atom_2 = mol.GetAtomWithIdx(f_group.H_alkyl.get_id())
                atom_1.SetAtomicNum(10)
                atom_2.SetAtomicNum(10)
        return mol

dashboard = LigninDashboard()
dashboard.app().show()

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