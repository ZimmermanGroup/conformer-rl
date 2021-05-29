import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
from torsionnet.utils import tfd_matrix

def _load_from_pickle(filename):
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data

def load_data_from_pickle(paths, indices=None):
    if not isinstance(paths, list):
        paths = [paths]

    if indices is None:
        indices = [f'test{i}' for i, x in enumerate(paths)]

    data = map(_load_from_pickle, paths)
    data =  list(data)

    final_data = {"indices": indices}
    for datum in data:
        for key, val in datum.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    final_data.setdefault(subkey, []).append(subval)
            else:
                final_data.setdefault(key, []).append(val)
    
    return final_data

def bar_plot_episodic(key, data):
    ax = sns.barplot(x=data["indices"], y=data[key])
    ax.set(xlabel='run', ylabel=key)
    return ax

def histogram_episodic(key, data):
    n = len(data["indices"])
    fig, axes = plt.subplots(ncols=n, figsize=(20, 8))

    if n == 1:
        sns.histplot(data=data[key][0], stat="probability", ax=axes)
        axes.set(xlabel=data['indices'][0])
    else:
        for i, index in enumerate(data["indices"]):
            sns.histplot(data=data[key][i], stat="probability", ax=axes[i])
            axes[i].set(xlabel=index)

    return fig, axes

def heatmap_episodic(key, data):
    pass

def tfd_heatmap(data):
    if not 'mol' in data:
        raise Exception('data dict must contain RDKit Mol object with \'mol\' key to generate tfd heatmap.')
    n = len(data["indices"])
    fig, axes = plt.subplots(ncols=n, figsize=(20, 8))
    if n == 1:
        sns.heatmap(data=tfd_matrix(data['mol'][0]), ax=axes)
        axes.set(xlabel=data['indices'][0])
    else:
        for i, index in enumerate(data["indices"]):
            sns.heatmap(data=tfd_matrix(data['mol'][i]), ax=axes[i])
            axes[i].set(xlabel=index)
    
    return fig, axes

def diversity_bar(data):
    diversity = []
    for molecules in data['mol']:
        diversity.append(np.sum(tfd_matrix(molecules)))
    ax = sns.barplot(x=data["indices"], y=diversity)
    ax.set(xlabel='run', ylabel='diversity metric')
    return ax


import py3Dmol

def MolTo3DView(mol, size=(300, 300), style="stick", surface=False, opacity=0.5):
    """Draw molecule in 3D
    
    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'carton')
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({style:{}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer