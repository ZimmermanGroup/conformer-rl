import numpy as np
import bisect
from rdkit.Chem import TorsionFingerprints
import rdkit.Chem.AllChem as Chem

from typing import Tuple


def get_conformer_energies(mol):
    energies = []
    Chem.MMFFSanitizeMolecule(mol)
    mmff_props = Chem.MMFFGetMoleculeProperties(mol)
    for conf in mol.GetConformers():
        ff = Chem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf.GetId())
        energy = ff.CalcEnergy()
        energies.append(energy)
    
    return np.asarray(energies, dtype=float)

def get_conformer_energy(mol, confId = None):
    if confId is None:
        confId = mol.GetNumConformers() - 1
    Chem.MMFFSanitizeMolecule(mol)
    mmff_props = Chem.MMFFGetMoleculeProperties(mol)
    ff = Chem.MMFFGetMoleculeForceField(mol, mmff_props, confId=confId)
    energy = ff.CalcEnergy()

    return energy

def prune_last_conformer(mol, tfd_thresh, energies):
    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol, energies

    idx = bisect.bisect(energies[:-1], energies[-1])
    tfd = TorsionFingerprints.GetTFDBetweenConformers(mol, range(0, mol.GetNumConformers() - 1), [mol.GetNumConformers() - 1], useWeights=False)
    tfd = np.array(tfd)

    # if lower energy conformer is within threshold, drop new conf
    if not np.all(tfd[:idx] >= tfd_thresh):
        energies = energies[:-1]
        mol.RemoveConformer(mol.GetNumConformers() - 1)
        return mol, energies
    else:
        keep = list(range(0, idx))
        keep.append(mol.GetNumConformers() - 1)
        keep += [x for x in range(idx, mol.GetNumConformers() - 1) if tfd[x] >= tfd_thresh]

        new = Chem.Mol(mol)
        new.RemoveAllConformers()
        for i in keep:
            conf = mol.GetConformer(i)
            new.AddConformer(conf, assignId=True)

        return new, [energies[i] for i in keep]

def prune_conformers(mol, tfd_thresh):
    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol

    energies = get_conformer_energies(mol)
    tfd = tfd_matrix(mol)
    sort = np.argsort(energies)  # sort by increasing energy
    keep = []  # always keep lowest-energy conformer
    discard = []

    for i in sort:
        this_tfd = tfd[i][np.asarray(keep, dtype=int)]
        # discard conformers within the tfd threshold
        if np.all(this_tfd >= tfd_thresh):
            keep.append(i)
        else:
            discard.append(i)

    # create a new molecule to hold the chosen conformers
    # this ensures proper conformer IDs and energy-based ordering
    new = Chem.Mol(mol)
    new.RemoveAllConformers()
    for i in keep:
        conf = mol.GetConformer(int(i))
        new.AddConformer(conf, assignId=True)

    return new

def tfd_matrix(mol):
    tfd = TorsionFingerprints.GetTFDMatrix(mol, useWeights=False)
    n = int(np.sqrt(len(tfd)*2))+1
    idx = np.tril_indices(n, k=-1, m=n)
    matrix = np.zeros((n,n))
    matrix[idx] = tfd
    matrix += np.transpose(matrix)
    return matrix

def calculate_normalizers(mol: Chem.Mol, num_confs: int = 200, pruning_thresh: float = 0.05) -> Tuple[float, float]:
    confslist = Chem.EmbedMultipleConfs(mol, numConfs = num_confs, useRandomCoords=True, numThreads=-1)
    if (len(confslist) < 1):
        raise Exception('Unable to embed molecule with conformer using rdkit')

    Chem.MMFFOptimizeMoleculeConfs(mol, maxIters=200, numThreads=-1)
    mol = prune_conformers(mol, pruning_thresh)

    energys = get_conformer_energies(mol)
    E0 = energys.min()
    Z0 = np.sum(np.exp(-(energys - E0)))

    mol.RemoveAllConformers()

    return E0, Z0