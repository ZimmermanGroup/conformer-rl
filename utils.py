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

import py3Dmol


def drawit(m, p, confId=-1):
    mb = Chem.MolToMolBlock(m, confId=confId)
    p.removeAllModels()
    p.addModel(mb, "sdf")
    p.setStyle({"stick":{}})
    p.setBackgroundColor("0xeeeeee")
    p.zoomTo()
    return p.show()

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

    def prune_conformers(self, mol, rmsd, heavy_atoms_only=True):
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

        new_rmsd = get_conformer_rmsd_fast(new, heavy_atoms_only=heavy_atoms_only)
        return new, new_rmsd

def get_conformer_rmsd_fast(mol, heavy_atoms_only=True):
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
    m2 = Chem.RemoveHs(mol)
    num_atoms = m2.GetNumAtoms()
    for i, ref_conf in enumerate(mol.GetConformers()):
        pbar.set_description("Calculating RMSDs of conformer %s" % i)
        for j, fit_conf in enumerate(mol.GetConformers()):
                if i >= j:
                        continue
    #           rmsd[i, j] = AllChem.GetBestRMS(mol, mol, ref_conf.GetId(),
    #                                       fit_conf.GetId())
                if not heavy_atoms_only:
                    rmsd[i, j] = AllChem.GetConformerRMS(mol, ref_conf.GetId(), fit_conf.GetId())
                else:
                    rmsd[i, j] = AllChem.GetConformerRMS(mol, ref_conf.GetId(), fit_conf.GetId(), atomIds=range(num_atoms))

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
    n = int(np.sqrt(len(arr)*2))+1
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

def enumerateTorsions(mol):
    torsionSmarts = '[!$(*#*)&!D1]~[!$(*#*)&!D1]'
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = mol.GetSubstructMatches(torsionQuery)
    torsionList = []
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = mol.GetBondBetweenAtoms(idx2, idx3)
        jAtom = mol.GetAtomWithIdx(idx2)
        kAtom = mol.GetAtomWithIdx(idx3)
        if (((jAtom.GetHybridization() != Chem.HybridizationType.SP2) and (jAtom.GetHybridization() != Chem.HybridizationType.SP3)) or ((kAtom.GetHybridization() != Chem.HybridizationType.SP2) and (kAtom.GetHybridization() != Chem.HybridizationType.SP3))):
            continue
        for b1 in jAtom.GetBonds():
            if (b1.GetIdx() == bond.GetIdx()):
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if ((b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx())):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if (idx4 == idx1):
                    continue
                torsionList.append((idx1, idx2, idx3, idx4))
    return torsionList