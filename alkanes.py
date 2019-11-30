from deepchem.utils import conformers, rdkit_util
from tqdm import tqdm
import mdtraj as md
import matplotlib.pyplot as plt
import nglview as nv
import numpy as np
from re import sub

from rdkit import Chem, DataStructs, RDConfig, rdBase
from rdkit import rdBase
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.Chem import Draw,PyMol,rdFMCS
from rdkit.Chem.Draw import IPythonConsole
# %alias_magic t timeit

import gym
from gym import spaces
from gym.envs.registration import registry, register, make, spec

from itertools import product

print('importing alkanes')
# %load funcs.py
# %load funcs.py
# %load funcs.py
# don't write functions directly on jupyter text editor (spaces are pasted wrong)

def load_from_sdf(sdf_file):
    """
    """
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False) #, strictParsing=False
    sdf_mols = [mol for mol in suppl]
    return sdf_mols

def load_trajectories(dcd_files, psf_files, stride=1):
    """
    Parameters
    ----------
    dcd_files: list of strings corresponding to dcd file paths
    psf_files: list of strings corresponding to psf file paths
    stride: step size (frames) of the Trajectory to load

    Returns
    -------
    trajs: list of Trajectory objects
    """
    trajs = []
    for i in range(len(dcd_files)):
        traj = md.load_dcd(dcd_files[i], psf_files[i], stride=stride)
        # align states onto first frame******
        traj.superpose(traj, frame=0)
        trajs.append(traj)
    return trajs

def add_conformers_to_molecule(mol, confs):
    """
    Parameters
    ----------
    mol: molecule to add conformers to
    confs: list of conformers that were loaded as Molecule types

    Returns
    -------
    mol : Molecule with added conformers
    """
    for i, conf in enumerate(confs):
        if conf == None or conf.GetNumAtoms() == 0:
                continue # skip empty
        # remove default conformer on Molecule
        mol.RemoveAllConformers()
        # get conformer to add (*if rerunning, reload the conformers again because the IDs have been changed)
        c = conf.GetConformer(id=0)
        c.SetId(i)
        cid = c.GetId()
#     add each conformer to original input molecule
        mol.AddConformer(c, assignId=False)
    return mol


class ConformerGeneratorCustom(conformers.ConformerGenerator):
    # pruneRmsThresh=-1 means no pruning done here
    # I don't use embed_molecule() because it does AddHs() & EmbedMultipleConfs()
    def __init__(self, *args, **kwargs):
        super(ConformerGeneratorCustom, self).__init__(*args, **kwargs)


    # add progress bar
    def minimize_conformers(self, mol):
        """
        Minimize molecule conformers.

        Parameters
        ----------
        mol : RDKit Mol
                Molecule.
        """
        pbar = tqdm(total=mol.GetNumConformers())
        # pct_prog = 100 / mol.GetNumConformers()
    # #     i = 0
        for conf in mol.GetConformers():
    #       i += 1
            # pbar.set_description("Minimizing %s" % i)
            ff = self.get_molecule_force_field(mol, conf_id=conf.GetId())
            ff.Minimize()
            pbar.update(1)
        pbar.close()

    def prune_conformers(self, mol, rmsd):
        """
        Prune conformers from a molecule using an RMSD threshold, starting
        with the lowest energy conformer.

        Parameters
        ----------
        mol : RDKit Mol
                Molecule.

        Returns
        -------
        new: A new RDKit Mol containing the chosen conformers, sorted by
                 increasing energy.
        new_rmsd: matrix of conformer-conformer RMSD
        """
        if self.rmsd_threshold < 0 or mol.GetNumConformers() <= 1:
            return mol
        energies = self.get_conformer_energies(mol)
    #     rmsd = get_conformer_rmsd_fast(mol)

        sort = np.argsort(energies)  # sort by increasing energy
        keep = []  # always keep lowest-energy conformer
        discard = []

        for i in sort:
            # always keep lowest-energy conformer
            if len(keep) == 0:
                keep.append(i)
                continue

            # discard conformers after max_conformers is reached
            if len(keep) >= self.max_conformers:
                discard.append(i)
                continue

            # get RMSD to selected conformers
            this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]

            # discard conformers within the RMSD threshold
            if np.all(this_rmsd >= self.rmsd_threshold):
                keep.append(i)
            else:
                discard.append(i)

        # create a new molecule to hold the chosen conformers
        # this ensures proper conformer IDs and energy-based ordering
        new = Chem.Mol(mol)
        new.RemoveAllConformers()
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        for i in keep:
            conf = mol.GetConformer(conf_ids[i])
            new.AddConformer(conf, assignId=True)

        new_rmsd = get_conformer_rmsd_fast(new)
        return new, new_rmsd

def get_conformer_rmsd_fast(mol):
    """
    Calculate conformer-conformer RMSD.

    Parameters
    ----------
    mol : RDKit Mol
            Molecule.
    """
    rmsd = np.zeros((mol.GetNumConformers(), mol.GetNumConformers()), dtype=float)
    pbar = tqdm(total=mol.GetNumConformers())
    # pct_prog = 100 / mol.GetNumConformers()
    for i, ref_conf in enumerate(mol.GetConformers()):
        pbar.set_description("Calculating RMSDs of conformer %s" % i)
        for j, fit_conf in enumerate(mol.GetConformers()):
                if i >= j:
                        continue
    #           rmsd[i, j] = AllChem.GetBestRMS(mol, mol, ref_conf.GetId(),
    #                                       fit_conf.GetId())
                rmsd[i, j] = AllChem.GetConformerRMS(mol, ref_conf.GetId(), fit_conf.GetId())
                rmsd[j, i] = rmsd[i, j]
        pbar.update(1)
    pbar.close()
    return rmsd

def get_lower_triangle(arr, get_symm_mat=False):
    # convert list to lower triangle mat
    n = int(np.sqrt(len(rms_arr)*2))+1
    idx = np.tril_indices(n, k=-1, m=n)
    lt_mat = np.zeros((n,n))
    lt_mat[idx] = arr
    if get_symm_mat == True:
        return lt_mat + np.transpose(lt_mat) # symmetric matrix
    return lt_mat


def ExplicitBitVect_to_array(bitvector):
    """
    input: bitvector as ExplicitBitVect type
    output: bitvector as numpy array
    """
    if not isinstance(bitvector, type(np.array([1]))) and not isinstance(bitvector, type(None)):
        bitstring = bitvector.ToBitString()
        intmap = map(int, bitstring)
        return np.array(list(intmap))
    elif isinstance(bitvector, type(np.array([1]))):
        return bitvector


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

# matrix reshaping
def symmetric_to_array(mat):
    # symmetric matrix to 1D array (good for Butina clustering in rdkit)
    """
    array is rmsmatrix = [ a,
                           b, c,
                           d, e, f,
                           g, h, i, j]
    """
    n = mat.shape[1]
    ix_lt = np.tril_indices(n, k=-1)
    return mat[ix_lt]

def array_to_lower_triangle(arr, get_symm=False):
    # convert list to lower triangle mat
    n = int(np.sqrt(len(rms_arr)*2))+1
    idx = np.tril_indices(n, k=-1, m=n)
    lt_mat = np.zeros((n,n))
    lt_mat[idx] = arr
    if get_symm == True:
        return lt_mat + np.transpose(lt_mat) # symmetric matrix
    return lt_mat

def save_xyz(m, file):
    """
    Writes m's atomic coordinates to an xyz file.

    Parameters
    ----------
    m: Molecule or Conformer
    file: filename with xyz extension
    """
    with open(file, 'w') as file:
        file.write(str(m.GetNumAtoms())+'\n')
        file.write('\n')

        if isinstance(m, Chem.rdchem.Mol):
            atoms = [atom.GetSymbol() for atom in m.GetAtoms()]
            xyz = np.round_(rdkit_util.get_xyz_from_mol(m), 8) # n_atoms X 3 numpy array
            for i in range(len(atoms)):
                file.write("{} {} {} {}\n".format(atoms[i], xyz[i, 0], xyz[i, 1], xyz[i, 2]))

        elif isinstance(m, Chem.rdchem.Conformer):
            atoms = [atom.GetSymbol() for atom in m.GetOwningMol().GetAtoms()]
            xyz = m.GetPositions() # n X 3
            for i in range(len(atoms)):
                file.write("{} {} {} {}\n".format(atoms[i], xyz[i, 0], xyz[i, 1], xyz[i, 2]))


def get_torsions_degs(mol):
    nonring, ring = TorsionFingerprints.CalculateTorsionLists(mol)
    conf = mol.GetConformer(id=0)
    tups = [atoms[0] for atoms, ang in nonring]
    degs = [Chem.rdMolTransforms.GetDihedralDeg(conf, *tup) for tup in tups]
    return degs                
                
def get_torsions(mol):
    degs = get_torsion_degs(mol)
    discdeg = [0] + [translate_to_discrete(deg) for deg in degs] + [4]
    return discdeg
    
def print_torsions(mol):
    nonring, ring = TorsionFingerprints.CalculateTorsionLists(mol)
    conf = mol.GetConformer(id=0)
    tups = [atoms[0] for atoms, ang in nonring]
    degs = [Chem.rdMolTransforms.GetDihedralDeg(conf, *tup) for tup in tups]
    print(degs)
    
def print_energy(mol):
    confgen = ConformerGeneratorCustom(max_conformers=1, 
                                 rmsd_threshold=None, 
                                 force_field='mmff',
                                 pool_multiplier=1)
    print(confgen.get_conformer_energies(mol)) 
    
def randomize_conformer(mol):
    nonring, ring = TorsionFingerprints.CalculateTorsionLists(mol)
    conf = mol.GetConformer(id=0)
    for tors in nonring:
        atoms, ang = tors
        tup = atoms[0]
        deg = Chem.rdMolTransforms.GetDihedralDeg(conf, *tup)
        Chem.rdMolTransforms.SetDihedralDeg(conf, *tup, deg + 120 * np.random.randint(-1, 2))

mols = load_from_sdf('./alkanes/alkanes.sdf') 
mols += load_from_sdf('./alkanes/pentane.sdf')
mols += load_from_sdf('./alkanes/heptane.sdf') 
mols += load_from_sdf('./alkanes/nonane.sdf') 
mols += load_from_sdf('./alkanes/11_alkane.sdf') 
mols += load_from_sdf('./alkanes/12_alkane.sdf') 
mols += load_from_sdf('./alkanes/14_alkane.sdf') 
mols += load_from_sdf('./alkanes/16_alkane.sdf') 
mols += load_from_sdf('./alkanes/18_alkane.sdf') 
mols += load_from_sdf('./alkanes/20_alkane.sdf') 

mols_by_rbn = {}

for mol in mols:
    m = Chem.rdmolops.AddHs(mol)
    AllChem.EmbedMolecule(m)
    nonring, ring = TorsionFingerprints.CalculateTorsionLists(m)
    conf = m.GetConformer(id=0)
    for tors in nonring:
        atoms, ang = tors
        tup = atoms[0]
        deg = Chem.rdMolTransforms.GetDihedralDeg(conf, *tup)
        Chem.rdMolTransforms.SetDihedralDeg(conf, *tup, 180.0)

    Chem.AllChem.MMFFOptimizeMolecule(m)

    atoms = m.GetNumAtoms()
    rbn = Chem.rdMolDescriptors.CalcNumRotatableBonds(m) - 2 
    print(rbn)
    mols_by_rbn[rbn] = m

confgen = ConformerGeneratorCustom(max_conformers=1, 
                 rmsd_threshold=None, 
                 force_field='mmff',
                 pool_multiplier=1)  

energy_max = {}
for num, mol in sorted(mols_by_rbn.items()):
    energy = np.exp(-1.0 * confgen.get_conformer_energies(mol)[0])
    energy_max[num] = energy
    
# Z = {
#     1: 202.747829286591,
#     2: 287.2264947637364,
#     3: 410.1948064928314,
#     4: 589.0199572719013,
#     5: 847.1997013690351,
#     6: 1219.977286030949,
#     7: 1757.8757380586885,
#     8: 2533.9561311348916,
#     9: 3653.522091596034,
#     11: 7599.921871806019,
#     13: 15819.841252757644,
# }

# top_n = {
#     1: 202.74911797108538,
#     2: 287.2265490857062,
#     3: 410.19434516277266,
#     4: 589.0305102196965,
#     5: 831.15924230415,
#     6: 1191.1514450692125,
#     7: 1646.2425377264608,
#     8: 2359.906685402908,
#     9: 3222.1225749490986,
#     11: 5747.164570622177,
#     13: 9400.403970830801,
# }

Z = {
    1: 298.2976919525584,
    2: 586.8618761962805,
    3: 1165.96340306337,
    4: 2335.3407079008293,
    5: 4691.896132512742,
    6: 9457.892895270266,
    7: 15099.886697178925,
}

top_n = {
    1 :298.2976919525584,
    2: 586.8618761962805,
    3: 1165.963395952183,
    4: 2333.1421795264077,
    5: 3572.350335700116,
    6: 6797.137366295856,
    7: 8645.893996411598,
}
    
def translate_to_discrete(deg):
    deg = round(deg)

    if deg == 60:
        return 1
    elif deg == 180 or deg == -180:
        return 2
    else:
        return 3 # for -60
    
def circular_format(deg):
    deg = round(deg)
    
    x = 0.5 * np.cos( (deg / 180.0) * np.pi) + 0.5
    y = 0.5 * np.sin( (deg / 180.0) * np.pi) + 0.5
    
    ans = [0, x, y]
    return ans

    
    
class AlkaneEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def _get_choice(self):
        if self.mode == 'train':
            choice = range(1, 18, 2)[np.random.randint(3)]
        elif self.mode == 'train_simple':
            choice = range(1, 18, 2)[0]
        elif self.mode == 'test':
            choice = 7
        elif self.mode == 'test_hard':
            choice = np.random.choice(range(1, 18, 2))
            
        return choice
          
    def _random_start(self):
        choice = self._get_choice()

        self.rbn = choice
        self.action_space = self.space_sets[choice]['action']
        self.observation_space = self.space_sets[choice]['obs']
        self.mol = self.space_sets[choice]['mol']
        self.max_gibbs = energy_max[choice]
        
        self.nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.conf = self.mol.GetConformer(id=0)     
                
        for tors in self.nonring:
            atoms, ang = tors
            tup = atoms[0]
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tup)
            Chem.rdMolTransforms.SetDihedralDeg(self.conf, *tup, deg + 120 * np.random.randint(-1, 2))       
        
    def _get_reward(self):
        return np.exp(-1.0 * confgen.get_conformer_energies(self.mol)[0]) / self.max_gibbs    

    def _get_obs(self):
        tups = [atoms[0] for atoms, ang in self.nonring]
        degs = [Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tup) for tup in tups]
        discdeg = [0] + [translate_to_discrete(deg) for deg in degs] + [4]
        
        return np.array(discdeg)
    
    def _load_spaces(self):
        self.space_sets = {}
        self.n_torsions = range(1, 18, 2)
        
        for n in Z.keys():
            self.space_sets[n] = {
                'action': spaces.MultiDiscrete([2] * (n + 2)), 
                'obs': spaces.MultiDiscrete([5] * (n+2)),
                'mol': self.mols_by_rbn[n],
            }
            
    def __init__(self, mode):
        super(AlkaneEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        
        self.mode = mode
        self.mols_by_rbn = mols_by_rbn
        self._load_spaces()
        self._random_start()

     
    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
                
        for idx, a in enumerate(action[1:-1]):
            tors = self.nonring[idx]
            atoms, ang = tors
            tup = atoms[0]
            
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tup)
            
            if a:
                Chem.rdMolTransforms.SetDihedralDeg(self.conf, *tup, deg + 120.0)     
    
        if self.rbn <= 3:
            done = self.current_step == 25
        elif self.rbn == 4 or self.rbn == 5:
            done = self.current_step == 50
        elif self.rbn == 6 or self.rbn == 7:
            done = self.current_step == 100
        else:
            done = self.current_step == 200

        
        obs = self._get_obs()
        rew = self._get_reward()
        done = self.current_step == 100
        
        print("max gibbs is ", self.max_gibbs)
        print("action is ", action)
        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)
        
        return obs, rew, done, {}
            
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self._random_start()
        obs = self._get_obs()
        print('reset called')
        print_torsions(self.mol)
        return obs
        
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print_torsions(self.mol)
        
class AlkaneWithoutReplacementEnv(AlkaneEnv):
    def _get_reward(self):
        obs = tuple(self._get_obs())
    
        if obs in self.seen:
            print('already seen')
            return 0
        else:
            self.seen.add(obs)
            print('Z is ', self.Z)
            print('gibbs ', np.exp(-1.0 * confgen.get_conformer_energies(self.mol)[0]))
            return 100 * np.exp(-1.0 * confgen.get_conformer_energies(self.mol)[0]) / self.Z    
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self._random_start()
        self.seen = set({})
        obs = self._get_obs()
        print('reset called')
        print_torsions(self.mol)
        return obs
    
    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
                
        for idx, a in enumerate(action[1:-1]):
            tors = self.nonring[idx]
            atoms, ang = tors
            tup = atoms[0]
            
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tup)
            
            if a:
                Chem.rdMolTransforms.SetDihedralDeg(self.conf, *tup, deg + 120.0)     
    
        if self.rbn <= 3:
            done = self.current_step == 25
        elif self.rbn == 4 or self.rbn == 5:
            done = self.current_step == 50
        elif self.rbn == 6 or self.rbn == 7:
            done = self.current_step == 100
        else:
            done = self.current_step == 200

        
        obs = self._get_obs()
        rew = self._get_reward()
        
        print("max gibbs is ", self.max_gibbs)
        print("action is ", action)
        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)
        
        return obs, rew, done, {}
                
    def _get_choice(self):
        if self.mode == 'train':
            choice = np.random.choice(range(4, 8))
        elif self.mode == 'train_simple':
            choice = 5
        elif self.mode == 'test':
            choice = 8
        elif self.mode == 'test_hard':
            choice = 13
        return choice
    
    def _random_start(self):
        super(AlkaneWithoutReplacementEnv, self)._random_start()
        self.Z = top_n[self.rbn]

class AlkaneDecoderEnv(AlkaneWithoutReplacementEnv):
    def _load_spaces(self):
        self.space_sets = {}
        self.n_torsions = range(1, 18, 2)
        
        for n in Z.keys():
            self.space_sets[n] = {
                'action': spaces.MultiDiscrete([5] * (n + 2)), 
                'obs': spaces.MultiDiscrete([5] * (n+2)),
                'mol': self.mols_by_rbn[n],
            }    
    
    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
                
        for idx, a in enumerate(action[1:-1]):
            tors = self.nonring[idx]
            atoms, ang = tors
            tup = atoms[0]
            
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tup)
            Chem.rdMolTransforms.SetDihedralDeg(self.conf, *tup, 60.0 + int(a)* 120.0)     

                    
            
            
            
            
        if self.rbn <= 3:
            done = self.current_step == 25
        elif self.rbn == 4 or self.rbn == 5:
            done = self.current_step == 50
        elif self.rbn == 6 or self.rbn == 7:
            done = self.current_step == 100
        else:
            done = self.current_step == 200

        
        obs = self._get_obs()
        rew = self._get_reward()
        
        print("max gibbs is ", self.max_gibbs)
        print("action is ", action)
        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)
        
        return obs, rew, done, {}    
    
print('glug glug')
class AlkaneConvolutionEnv(AlkaneDecoderEnv):
    def _get_reward(self):
        obs = tuple(get_torsions(self.mol))
    
        if obs in self.seen:
            print('already seen')
            return 0
        else:
            self.seen.add(obs)
            print('Z is ', self.Z)
            print('gibbs ', np.exp(-1.0 * confgen.get_conformer_energies(self.mol)[0]))
            return 100 * np.exp(-1.0 * confgen.get_conformer_energies(self.mol)[0]) / self.Z   
        
    def _get_obs(self):
        
        dm = AllChem.Get3DDistanceMatrix(self.mol)
        dm = dm / dm.max()
        adj = Chem.rdmolops.GetAdjacencyMatrix(self.mol).astype(dm.dtype)
        stacked = np.stack([adj, dm])

        return stacked   
     
class AlkaneConvolutionEnv(AlkaneDecoderEnv):
    def _get_reward(self):
        obs = tuple(get_torsions(self.mol))
    
        if obs in self.seen:
            print('already seen')
            return 0
        else:
            self.seen.add(obs)
            print('Z is ', self.Z)
            print('gibbs ', np.exp(-1.0 * confgen.get_conformer_energies(self.mol)[0]))
            return 100 * np.exp(-1.0 * confgen.get_conformer_energies(self.mol)[0]) / self.Z   
        
    def _get_obs(self):
        
        dm = AllChem.Get3DDistanceMatrix(self.mol)
        dm = dm / dm.max()
        adj = Chem.rdmolops.GetAdjacencyMatrix(self.mol).astype(dm.dtype)
        stacked = np.stack([adj, dm])

        return stacked          
                    

        
        
# class AlkaneCircularSpaceEnv(AlkaneEnv):
#     def _load_spaces(self):
#         self.space_sets = {}
#         self.n_torsions = range(1, 18, 2)
        
#         for n in self.n_torsions:
#             self.space_sets[n] = {
#                 'action': spaces.MultiDiscrete([2] * (n + 2)), 
#                 'obs': spaces.Box(low=0.0, high=1.0, shape=(3, (n+2))),
#                 'mol': self.mols_by_rbn[n]
#             }
            
#     def _get_obs(self):
#         tups = [atoms[0] for atoms, ang in self.nonring]
#         degs = [Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tup) for tup in tups]
#         circ = [[1,0,0]] + [circular_format(deg) for deg in degs] + [[1,0,0]]
        
#         return np.array(circ)
