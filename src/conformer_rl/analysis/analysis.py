"""
Analysis
========

Functions for analyzing and visualizing (in Jupyter/IPython notebook) logged
environment data. The functions for visualizations here provide a basic set of functionality
to guide users in understanding the format of the logged environment data. 
Users are encouraged to generate their own plots and visualizations based on their specific needs.
"""

import pickle
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from rdkit import Chem
from conformer_rl.utils import tfd_matrix
import py3Dmol
import logging

from typing import Any, List, Optional, Tuple

def _load_from_pickle(filename: str) -> Any:
    """Loads an object from a .pickle file.
    """
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data

def load_data_from_pickle(paths: List[str], indices: Optional[List[str]]=None) -> dict:
    """Loads saved pickled environment data from multiple runs into a combined data dict.

    Parameters
    ----------
    paths : list of str
        List of paths to .pickle files corresponding to the environment data from the runs of interest.
    indices : list of str, optional
        Specifies custom indices/labels to be displayed in generated Seaborn graphs for each run. Should be the
        same length as `paths`. If not specified, the labels default to ``test0, test1, test2, ...``.

    Returns
    -------
    dict mapping from str to list
        The str corresponds to the key for the data in the original pickled dict object. The list contains
        the data for each of the environment data sets specified in `paths`, in the same order they were
        given in `paths`.

    Notes
    -----
    The ``.pickle`` files specified by `paths` should be dumped directly by :class:`~conformer_rl.logging.env_logger.EnvLogger`,
    and should correspond to a single evaluation episode.
    See :meth:`conformer_rl.logging.env_logger.EnvLogger.save_episode` for more details on the dumped format.

    An example of how the function operates: Suppose that our paths are::

        ['data1.pickle', 'data2.pickle', 'data3.pickle']

    And each pickle object contains corresponding data::

        data1 = {
            'total_rewards': data1_total_rewards,
            'mol': data1_molecule,
            'rewards': [data1_step1_rewards, data1_step2_rewards, data1_step3_rewards, data1_step4_rewards]
        }
        data2 = {
            'total_rewards': data2_total_rewards,
            'mol': data2_molecule,
            'rewards': [data2_step1_rewards, data2_step2_rewards, data2_step3_rewards, data2_step4_rewards]
        }
        data3 = {
            'total_rewards': data3_total_rewards,
            'mol': data3_molecule,
            'rewards': [data3_step1_rewards, data3_step2_rewards, data3_step3_rewards, data3_step4_rewards]
        }
    
    Suppose that data1 corresponds to some eval data obtained from training with the PPO agent, data2 was obtained
    from the PPORecurrent agent, and data3 was obtained from training with the A2C agent. Then we can input custom `indices`
    to help us understand each dataset better::

        indices = ['PPO', 'PPO_recurrent', 'A2C']

    Given these data and indices, :func:`load_data_from_pickle` would return the following dict::

        {
            'indices': ['PPO', 'PPO_recurrent', 'A2C'],
            'total_rewards': [
                data1_total_rewards,
                data2_total_rewards,
                data3_total_rewards
            ],
            'mol': [
                data1_molecule,
                data2_molecule,
                data3_molecule
            ],
            'rewards': [
                [data1_step1_rewards, data1_step2_rewards, data1_step3_rewards, data1_step4_rewards],
                [data2_step1_rewards, data2_step2_rewards, data2_step3_rewards, data2_step4_rewards],
                [data3_step1_rewards, data3_step2_rewards, data3_step3_rewards, data3_step4_rewards]
            ],
        }
    
    This format consolidates all the data into a single dict and is compatible with the other visualization
    functions in this module. Furthermore, it is also easy to convert a dict of this format into a Pandas dataframe
    or other tabular formats if needed.
    """

    if not isinstance(paths, list):
        paths = [paths]

    if indices is None:
        indices = [f'test{i}' for i, x in enumerate(paths)]

    data = map(_load_from_pickle, paths)
    data =  list(data)

    final_data = {"indices": indices}
    for datum in data:
        if 'step_data' in datum:
            datum.update(datum['step_data'])
            del datum['step_data']
        for key, val in datum.items():
            final_data.setdefault(key, []).append(val)
            
    return final_data

def list_keys(data: dict) -> List[str]:
    """Return a list of all keys in a dict.

    Parameters
    ----------
    data : dict
        The dictionary to retrieve keys from.
    """
    return list(key for key, val in data.items())

def bar_plot_episodic(key: str, data: dict) -> matplotlib.axes.Axes:
    """Plots a bar plot comparing a scalar value across all episodes loaded in `data`.

    Parameters
    ----------
    key : str
        The key for the values to be compared across all data sets/episodes.
    data : dict
        Data dictionary generated by :meth:`load_data_from_pickle`.
    """
    ax = sns.barplot(x=data["indices"], y=data[key])
    ax.set(xlabel='run', ylabel=key)
    return ax

def histogram_select_episodes(key: str, data: dict, episodes: List[int]=None, binwidth: float=10, figsize: Tuple[float, float]=(8., 6.)) -> matplotlib.axes.Axes:
    """Plots a single histogram where data for each episode in `episodes` are overlayed.

    Parameters
    ----------
    key : str
        The key for the values to be compared across all data sets/episodes.
    data : dict
        Data dictionary generated by :meth:`load_data_from_pickle`.
    episodes : list of int, optional
        Specifies the indices in `data` for the episodes to be shown. If not specified,
        all episodes are shown.
    binwidth: float
        The width of each bin in the histogram.
    figsize: 2-tuple of float
        Specifies the size of the plot.
    """
    if episodes is None:
        episodes = list(range(len(data['indices'])))
    fig, axes = plt.subplots()
    input_data = {data["indices"][i]: data[key][i] for i in episodes}
    sns.histplot(data=input_data, binwidth=binwidth, ax=axes)
    axes.set(xlabel=key)

    return fig, axes

def histogram_episodic(key: str, data: dict, binwidth: int=10, figsize: Tuple[float, float]=(8., 6.)) -> matplotlib.axes.Axes:
    """Plots histogram on separate axes for each of the episode data sets in `data`.

    Parameters
    ----------
    key : str
        The key for the values to be compared across all data sets/episodes.
    data : dict
        Data dictionary generated by :meth:`load_data_from_pickle`.
    binwidth: float
        The width of each bin in the histogram.
    figsize: 2-tuple of float
        Specifies the size of the plot.
    """
    n = len(data["indices"])
    fig, axes = plt.subplots(nrows=n, figsize=figsize)

    if n == 1:
        sns.histplot(data={data['indices'][0]: data[key][0]}, binwidth=binwidth, ax=axes)
        axes.set(xlabel=key)
    else:
        for i, index in enumerate(data["indices"]):
            sns.histplot(data={index: data[key][i]}, binwidth=binwidth, ax=axes[i])
            axes[i].set(xlabel=key)

    return fig, axes

def heatmap_episodic(key: str, data: dict, figsize: Tuple[float, float]=(8., 6.)) -> matplotlib.axes.Axes:
    """Plots heatmap(s) for matrix data corresponding to `key` across all episodes
    loaded in `data`.

    Parameters
    ----------
    key : str
        The key for the values to be compared across all data sets/episodes.
    data : dict
        Data dictionary generated by :meth:`load_data_from_pickle`.
    figsize: 2-tuple of float
        Specifies the size of the plot.
    """
    n = len(data["indices"])
    fig, axes = plt.subplots(nrows=n, figsize=figsize)
    if n == 1:
        sns.heatmap(data=data[key][0], ax=axes)
        axes.set(xlabel=data['indices'][0])
    else:
        for i, index in enumerate(data["indices"]):
            sns.heatmap(data=data[key][i], ax=axes[i])
            axes[i].set(xlabel=index)
    
    return fig, axes

def calculate_tfd(data: str) -> None:
    """Updates data with the TFD (Torsion Fingerprint Deviation) matrix (with key 'tfd_matrix') and sum of the TFD matrix
    (with key 'tfd_total') for the molecule conformers across each episode loaded in `data`.

    Parameters
    ----------
    data : dict
        Data dictionary generated by :meth:`load_data_from_pickle`.
    """
    if not 'mol' in data:
        raise Exception('data dict must contain RDKit Mol object with \'mol\' key to generate tfd matrix.')
    if 'tfd_matrix' in data or 'tfd_total' in data:
        logging.info("tfd matrix already exists, recalculating...")
        data.pop('tfd_matrix')
        data.pop('tfd_total')
    for mol in data['mol']:
        matrix = tfd_matrix(mol)
        data.setdefault('tfd_matrix', []).append(matrix)
        data.setdefault('tfd_total', []).append(np.sum(matrix))

def drawConformer(mol: Chem.Mol, confId: int=-1, size: Tuple[int, int]=(300, 300), style: str="stick") -> py3Dmol.view:
    """Displays interactive 3-dimensional representation of specified conformer.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule containing the conformer to be displayed.
    confId : int
        The ID of the conformer to be displayed.
    size : Tuple[int, int]
        The size of the display (width, height).
    style: str
        The drawing style for displaying the molecule. Can be sphere, stick, line, cross, cartoon, and surface.
    """
    block = Chem.MolToMolBlock(mol, confId=confId)
    view = py3Dmol.view(width=size[0], height=size[1])
    view.addModel(block, 'mol')
    view.setStyle({style : {}})
    view.zoomTo()
    return view

def drawConformer_episodic(data: dict, confIds: List[int], size: Tuple[int, int]=(300, 300), style: str="stick") -> py3Dmol.view:
    """Displays a specified conformer for each episode loaded in `data`.

    Parameters
    ----------
    data : dict from string to list
        Contains the loaded episode information. 'mol' must be a key in data and the corresponding list must contain
        RDKit Mol objects.
    confIds : list of int
        The indices for the conformers to be displayed (for each episode loaded in data).
    size : Tuple[int, int]
        The size of the display for each individual molecule (width, height).
    style: str
        The drawing style for displaying the molecule. Can be sphere, stick, line, cross, cartoon, and surface.
    """
    n = len(data['mol'])
    view = py3Dmol.view(width=size[0]*n, height=size[0]*n, linked=False, viewergrid=(n, 1))
    for i in range(n):
        block = Chem.MolToMolBlock(data['mol'][i], confId=confIds[i])
        view.addModel(block, 'mol', viewer=(i, 0))
        view.setStyle({style:{}}, viewer=(i, 0))
    view.zoomTo()
    return view