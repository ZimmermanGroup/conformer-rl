import sys, os, pickle
# print(os.getcwd())
# sys.path.append("./Python/tests")
# print(sys.path)
from conformer_generator import *
from utility_functions import *

if __name__ == '__main__':
    # files from IQMol
    scratch_path, scratch_dirs, scratch_files = os.walk("/Users/exequielpunzalan/Desktop/iqmol_scratch").__next__()
    print("Files saved in {}".format(scratch_path))
    # print(scratch_path, scratch_dirs, scratch_files)
    input_file = [scratch_path+"/input/input.mol"] # what the user starts working with
    mol_files = [scratch_path+"/"+f for f in scratch_files if f[-4:] == '.mol'] # only get mol files
    # print(mol_files)
    num_files = len(mol_files)
    # print(num_files)

    # .mol files --> 1 .sdf file
    molfile_to_sdf(mol_files, outfile=scratch_path+"/conformers.sdf")
    # read in SDF to RDKit
    mols_out = load_from_sdf(scratch_path + "/conformers.sdf")
    # print([mol.GetNumAtoms() for mol in mols_out])

    # pruning/uniqueness for Molecule
    confgen = ConformerGeneratorCustom(max_conformers=num_files,
                                       rmsd_threshold=float(sys.argv[1]),
                                       tfd_threshold=float(sys.argv[2]),
                                       force_field=sys.argv[3],
                                       pool_multiplier=1)

    # assuming all files are the same molecule,
    # consolidate all Molecules into 1 RDKit Molecule such that they become RDKit Conformers
    input_mol_sdf = molfile_to_sdf(input_file, outfile=scratch_path+"/input/input.sdf")
    input_mol = load_from_sdf(scratch_path + "/input/input.sdf")[0]
    input_mol.RemoveAllConformers() # remove input Conformer object
    mol = confgen.add_conformers_as_Molecules_to_Molecule(input_mol, confs=mols_out)

    # pruning
    rmsd = ConformerGeneratorCustom.get_conformer_rmsd_fast(mol)
    pruned_mol_rmsd, pruned_rmsd = confgen.prune_conformers(mol, rmsd, measure="rmsd")
    tfd = ConformerGeneratorCustom.get_tfd_matrix(mol)
    pruned_mol_tfd, pruned_tfd = confgen.prune_conformers(mol, tfd, measure="tfd")

    # partition function
    energies_rmsd = confgen.get_conformer_energies(pruned_mol_rmsd)
    energies_tfd = confgen.get_conformer_energies(pruned_mol_tfd)
    pf_rmsd = np.sum(np.exp(-energies_rmsd/0.593))
    pf_tfd = np.sum(np.exp(-energies_tfd/0.593)) # RT = 0.593 kcal/mol
    prob_confs_rmsd = [np.exp(-e/0.593)/pf_rmsd for e in energies_rmsd]
    prob_confs_tfd = [np.exp(-e/0.593)/pf_tfd for e in energies_tfd]
    # free energy score under uniqueness constraint
    kT = 4.11e-21/1000 # kJ
    F_rmsd = -kT * np.log(pf_rmsd)
    F_tfd = -kT * np.log(pf_tfd)
    print("free energy score (RMSD pruning): {} kJ with {} conformers".format(F_rmsd, pruned_mol_rmsd.GetNumConformers()))
    print("free energy score (TFD pruning): {} kJ {} conformers".format(F_tfd, pruned_mol_tfd.GetNumConformers()))

    # saving
    mols_save_dir = scratch_path + "/rdkit_mols/"
    os.makedirs(mols_save_dir, exist_ok=True)

    with open(mols_save_dir + "pruned_mol_{}rmsd.pkl".format(confgen.rmsd_threshold), 'wb') as pickle_file:
        pickle.dump(pruned_mol_rmsd, pickle_file)
    with open(mols_save_dir + "pruned_mol_{}tfd.pkl".format(confgen.tfd_threshold), 'wb') as pickle_file:
        pickle.dump(pruned_mol_tfd, pickle_file)
