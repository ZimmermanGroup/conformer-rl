import pickle
from rdkit import Chem

from torsionnet.utils import mkdir

class EvalLogger:
    def __init__(self, dir):
        self.base_dir = dir
        self.step_data = {}
        self.episode_data = {}

    def reset(self):
        self.step_data = {}
        self.episode_data = {}

    def log_step(self, step_data):
        for key, val in step_data.items():
            if key in self.step_data:
                self.step_data[key].append(val)
            else:
                self.step_data[key] = [val]

    def log_episode(self, episode_data):
        self.episode_data.update(episode_data)
        self.episode_data["step_data"] = self.step_data

    def dump_episode(self, agent_steps, episode=0, save_molecules=False):
        path = self.base_dir + f'/step_{agent_steps}/ep_{episode}'
        mkdir(path)
        filename = path + '/data.pickle'
        outfile = open(filename, 'w+b')
        pickle.dump(self.episode_data, outfile)
        outfile.close()

        if save_molecules and "molecule" in self.step_data:
            for index, mol in enumerate(self.step_data["molecule"]):
                Chem.MolToMolFile(mol, path + f'/step_{index}.mol')

        self.reset()