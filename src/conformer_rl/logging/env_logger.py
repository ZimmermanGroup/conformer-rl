"""
Env_logger
==========
"""

import pickle
from rdkit import Chem
from conformer_rl.utils import mkdir
from typing import Any

class EnvLogger:
    """Used by the agent for logging metrics produced by the environment, for example, observations, rewards, renders, etc.
    Supports saving data to pickle and saving molecules as .mol files.

    Parameters
    ----------
    tag : str
        Unique tag for identifying the logging session.
    dir : str
        Path to root directory for where logging results should be saved.


    Attributes
    ----------
    step_data : dict
        Used for storing data for every step of a single episode. 
        Maps from strings to lists, where each index of the list corresponds
        to the data for that corresponding step within the episode.
    episode_data : dict 
        Used for storing information for a single episode. Used to store the self.step_data
        for that episode and metadata global to the entire episode.
    cache : dict 
        Used for storing data across several episodes. Maps from strings to lists, where each
        index of the list corresponds to an episode.
    """
    def __init__(self, tag: str, dir: str = "data"):
        self.dir = dir
        mkdir(dir)
        self.tag = tag
        self.step_data = {}
        self.episode_data = {}
        self.cache = {}

    def clear_data(self) -> None:
        """Resets the logger.
        """
        self.cache = {}
        self.episode_data = {}
        self.step_data = {}

    def clear_episode(self) -> None:
        """Clears episode and step data.
        """
        self.step_data = {}
        self.episode_data = {}

    def log_step_item(self, key: str, val: Any) -> None:
        """Logs a single key value pair for current step.

        If an existing key is found, the value is appended
        to the list associated with that key.

        Parameters
        ----------
        key : str
            the key for the data to be added.
        val : Any
            the value of the data to be added.
        """
        if key in self.step_data:
            self.step_data[key].append(val)
        else:
            self.step_data[key] = [val]

    def log_step(self, step_data: dict) -> None:
        """Logs each key-value pair for current step.

        If an existing key is found, the value is appended
        to the list associated with that key.

        Parameters
        ----------
        step_data : dict
            Contains key-value pairs to be logged.
        """
        for key, val in step_data.items():
            self.log_step_item(key, val)

    def log_episode_item(self, key: str, value: Any) -> None:
        """Logs a single key-value pair to the per-episode data.
        """
        self.episode_data[key] = value

    def log_episode(self, episode_data: dict) -> None:
        """Logs each key-value pair to the per-episode data.

        Also adds `step_data` to the per-episode data with corresponding key
        'step_data'. Existing keys will be overwritten.

        Parameters
        ----------
        episode_data : dict
            Contains key-value pairs to be logged.
        """
        self.episode_data.update(episode_data)
        self.episode_data["step_data"] = self.step_data

    def save_episode(self, subdir: str, save_pickle: bool = True, save_molecules: bool = False, save_cache: bool = False) -> None:
        """Saves current episode_data with options for dumping to pickle file,
        saving data to a cache dict, and saving the rdkit molecules as .mol files.
        Clears the current episode and step data.

        Parameters
        ----------
        subdir : str
            The directory for episode data to be saved (relative to self.dir)
        save_pickle : bool
            If True, dumps episode_data as a .pickle file.
        save_molecules : bool
            If True, and 'molecule' key exists in step_data, dumps each molecule generated 
            throughout the episode as a .mol file uniquely named by the step number.
        save_cache : bool
            If True, episode data is cached to self.data.
        """
        path = self.dir + '/' +  'env_data' + '/' + self.tag + '/' + subdir
        mkdir(path)
        filename = path + '/' +  'data.pickle'

        if save_pickle:
            outfile = open(filename, 'w+b')
            pickle.dump(self.episode_data, outfile)
            outfile.close()

        if save_molecules and 'mol' in self.episode_data:
            mol = self.episode_data['mol']
            for i in range(mol.GetNumConformers()):
                Chem.MolToMolFile(mol, filename=path + '/' +  f'step_{i}.mol', confId=i)

        if save_cache:
            self._add_to_cache(self.episode_data)

        self.clear_episode()

    def _add_to_cache(self, data:dict) -> None:
        """Logs each key-value pair in data to self.cache.
        If an existing key is found, the value is appended
        to the list associated with that key.

        Parameters
        ----------
        data : dict
            contains key-value pairs to be logged.
        """
        for key, val in data.items():    
            if key in self.cache:
                self.cache[key].append(val)
            else:
                self.cache[key] = [val]