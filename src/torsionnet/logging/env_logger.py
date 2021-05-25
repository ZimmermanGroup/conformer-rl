import pickle
from rdkit import Chem
from torsionnet.utils import mkdir
from typing import Any

class EnvLogger:
    """
    Used by the agent for logging metrics produced by the environment.
    E.g., observations, rewards, renders, etc.
    Supports saving data to pickle and saving molecules as .mol files.

    ...

    Attributes
    ----------
    self.step_data: dict used for storing data for each and every step of a single episode.
    Maps from strings to lists, where each index of the list corresponds to the data for that
    corresponding step within the episode.
    self.episode_data: dict for storing information for a single episode. Used to store the self.step_data
    for that episode and metadata global to the entire episode.
    self.cache: dict used for storing data across several episodes. Maps from strings to lists, where each
    index of the list corresponds to an episode.
    """
    def __init__(self, tag: str, dir: str = "data"):
        """
        Parameters
        ----------
        dir: directory for saving log files (relative to working directory)
        """
        self.dir = dir
        mkdir(dir)
        self.tag = tag
        self.step_data = {}
        self.episode_data = {}
        self.cache = {}

    def clear_data(self) -> None:
        """
        Resets the logger.
        """
        self.cache = {}
        self.episode_data = {}
        self.step_data = {}

    def clear_episode(self) -> None:
        """
        Clears episode and step data.
        """
        self.step_data = {}
        self.episode_data = {}

    def log_step_item(self, key: str, val: Any) -> None:
        """
        Logs a single key value pair for current step.
        If an existing key is found, the value is appended
        to the list associated with that key.

        Parameters
        ----------
        key: the key for the data to be added.
        val: the value of the data to be added.
        """
        if key in self.step_data:
            self.step_data[key].append(val)
        else:
            self.step_data[key] = [val]

    def log_step_molecule(self, mol: Chem.Mol) -> None:
        """
        Logs a single rdkit molecule for current step.
        The key for the molecule will be 'molecule'.

        Parameters
        ----------
        mol: the rdkit molecule to be logged.
        """
        self.log_step_item(key = "molecule", val = mol)

    def log_step(self, step_data: dict) -> None:
        """
        Logs each key-value pair for current step.
        If an existing key is found, the value is appended
        to the list associated with that key.

        Parameters
        ----------
        step_data: dictionary containing key-value pairs to be logged.
        """
        for key, val in step_data.items():
            self.log_step_item(key, val)

    def log_episode_item(self, key: str, value: Any) -> None:
        self.episode_data[key] = value

    def log_episode(self, episode_data: dict) -> None:
        """
        Logs each key-value pair in episode data to a per-episode dict.
        Also includes the current step_data to the per-episode dict corresponding key
        step_data. Existing keys will be overwritten.

        Parameters
        ----------
        episode_data: dictionary containing key-value pairs to be logged.
        """
        self.episode_data.update(episode_data)
        self.episode_data["step_data"] = self.step_data

    def save_episode(self, subdir: str, save_pickle: bool = True, save_molecules: bool = False, save_cache: bool = False) -> None:
        """
        Stores current episode_data, with options for dumping to pickle file,
        saving data to a cache dict, and saving the rdkit molecules as .mol files.
        Clears the current episode and step data.

        Parameters
        ----------
        subdir: string representing the directory for episode data to be saved (relative to self.dir)
        save_pickle: if True, dumps episode_data as a .pickle file.
        save_molecules: if True, and 'molecule' key exists in step_data, dumps each molecule generated
        throughout the episode as a .mol file uniquely named by the step number.
        save_cache: if True, episode data is cached to self.data.
        """
        path = self.dir + '/' +  'env_data' + '/' + self.tag + '/' + subdir
        mkdir(path)
        filename = path + '/' +  'data.pickle'

        if save_pickle:
            outfile = open(filename, 'w+b')
            pickle.dump(self.episode_data, outfile)
            outfile.close()

        if save_molecules and "molecule" in self.step_data:
            for index, mol in enumerate(self.step_data["molecule"]):
                Chem.MolToMolFile(mol, path + '/' +  f'step_{index}.mol')

        if save_cache:
            self._add_to_cache(self.episode_data)

        self.clear_episode()

    def _add_to_cache(self, data:dict) -> None:
        """
        Logs each key-value pair in data to self.cache.
        If an existing key is found, the value is appended
        to the list associated with that key.

        Parameters
        ----------
        data: dictionary containing key-value pairs to be logged.
        """
        for key, val in data.items():    
            if key in self.cache:
                self.cache[key].append(val)
            else:
                self.cache[key] = [val]