from deepchem.utils import conformers, rdkit_util
from tqdm import tqdm
import copy
import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
import os
# import seaborn as sns

from rdkit import Chem, RDConfig, rdBase
from rdkit import rdBase
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.Chem import Draw, PyMol, rdFMCS
from rdkit.Chem.Draw import IPythonConsole

class ConformerGeneratorCustom(conformers.ConformerGenerator):
    # pruneRmsThresh=-1 means no pruning done here
    # I don't use embed_molecule() because it uses AddHs() & EmbedMultipleConfs()
    def __init__(self, tfd_threshold, **kwargs):
        self.tfd_threshold = tfd_threshold
        super(ConformerGeneratorCustom, self).__init__(**kwargs)

    # added progress bar
    def minimize_conformers(self, mol):
        """
        Minimize molecule conformers.

        Parameters
        ----------
        mol : RDKit Molecule
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

    def prune_conformers(self, mol, measure_mat, measure="rmsd"):
        """
        • Prune conformers from a molecule using a RMSD/TFD cutoff (lit: "TFD>0.2 practically relevant"),
        starting with the lowest energy conformer.
        • changes from deepchem version
            -added parameters: measure_mat, measure
            -now returns new Molecule and pruned RMSD/TFD array

        Parameters
        ----------
        mol : RDKit Molecule.
        measure_mat: RMSD/TFD matrix of pre-pruned conformers
        measure: string of which measure to use

        Returns
        ----------
        new_mol: RDKit Molecule with pruned conformers
        icRMSD matrix or icTFD array of pruned conformers
        """
        if measure == "rmsd":
            if self.rmsd_threshold < 0 or mol.GetNumConformers() <= 1:
                return mol, measure_mat
            rmsd = measure_mat
        elif measure =="tfd":
            if self.tfd_threshold < 0 or mol.GetNumConformers() <= 1:
                return mol, measure_mat
            tfd = measure_mat
        energies = self.get_conformer_energies(mol)

        sort = np.argsort(energies)  # sort by increasing energy
        keep = []  # always keep lowest-energy conformer
        discard = []

        print("Pruning conformers...")
        if measure == "rmsd":
            for i in sort:
                # always keep lowest-energy conformer
                if len(keep) == 0:
                    keep.append(i)
                    continue
                # discard conformers after max_conformers is reached
                if len(keep) >= self.max_conformers:
                    discard.append(i)
                    continue
                # get TFD to selected conformers
                this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]
                # discard conformers within the RMSD threshold
                if np.all(this_rmsd >= self.rmsd_threshold):
                    keep.append(i)
                else:
                    discard.append(i)
        elif measure == "tfd":
            for i in sort:
                # always keep lowest-energy conformer
                if len(keep) == 0:
                    keep.append(i)
                    continue
                # discard conformers after max_conformers is reached
                if len(keep) >= self.max_conformers:
                    discard.append(i)
                    continue
                # get TFD to selected conformers
                this_tfd = tfd[i][np.asarray(keep, dtype=int)]
                # discard conformers within the TFD threshold
                if np.all(this_tfd >= self.tfd_threshold):
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
        # compute icRMSD/icTFD of pruned conformers (TODO: save index of pruned conformers so no need to recompute)
        if measure == "rmsd":
            print("Computing RMSD matrix of pruned conformers...")
            new_rmsd = ConformerGeneratorCustom.get_conformer_rmsd_fast(new)
            print("Done.")
            return new, new_rmsd
        if measure == "tfd":
            print("Computing TFD array of pruned conformers...")
            # new_tfd_arr = Chem.TorsionFingerprints.GetTFDMatrix(new)
            # print("Done.")
            # return new, new_tfd_arr
            new_tfd = ConformerGeneratorCustom.get_tfd_matrix(new)
            print("Done")
            return new, new_tfd

    @staticmethod
    def get_conformer_rmsd_fast(mol):
        """
        Calculate conformer-conformer RMSD with progress bar.

        Parameters
        ----------
        mol : RDKit Molecule

        Returns
        -------
        rmsd: n X n numpy array where n is the number of conformers
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

    @staticmethod
    def get_tfd_matrix(mol):
    """
    Calculate conformer-conformer torsion fingerprint deviation (TFD) matrix

    Parameters
    ----------
    mol : RDKit Molecule

    Returns
    -------
    nXn numpy array where n is the number of conformers
    """
    tfd_arr = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev='equal', symmRadius=0, ignoreColinearBonds=True)
    tfd_lt = array_to_lower_triangle(tfd_arr)
    return tfd_lt + np.transpose(tfd_lt)

    @staticmethod
    def add_conformers_as_Molecules_to_Molecule(mol, confs):
        """
        Parameters
        ----------
        mol: Molecule to add conformers to
        confs: list of RDKit Molecules that we want to add as RDKit Conformers to mol

        Returns
        -------
        mol : Molecule with added conformers
        """
        # mol.RemoveAllConformers()
        for i, conf in enumerate(confs):
            if conf == None or conf.GetNumAtoms() == 0:
                    continue # skip empty
            # get conformer to add (*if rerunning, reload the conformers again because the IDs have been changed)
            c = conf.GetConformer(id=0)
            # c.SetId(i)
            # cid = c.GetId()
        # add each conformer to original input molecule
            mol.AddConformer(c, assignId=False)
        return mol

    @staticmethod
    def add_conformers_to_molecule(new_mol, old_mol, cids):
        """
        Parameters
        ----------
        new_mol: Molecule to add conformers to
        old_mol: Molecule to get conformers from
        cids: list of conformer ids in old_mol that we want to add to new_mol
        Returns
        -------
        new_mol: new_mol Molecule and with added conformers
        """
        confs = [old_mol.GetConformer(id=id) for id in cids]
        for conf in confs:
            new_mol.AddConformer(conf, assignId=False)
        return new_mol
