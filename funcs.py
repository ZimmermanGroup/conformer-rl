# %load funcs.py
# don't write functions directly here (spaces are pasted wrong)

def load_from_sdf(sdf_file):
    # returns a list of Molecules
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False) #, strictParsing=False
    sdf_mols = [mol for mol in suppl]
#     print(type(suppl))
#     print(type(sdf_mols))
    print("Molecules loaded.")
    return sdf_mols

def load_trajectories(dcd_files, psf_files, stride=1):
    """
    Parameters
    ----------
    dcd_files
    psf_files
    stride
    
    Returns
    -------
    trajs
    """
    trajs = []
    for i in range(len(dcd_files)):
        traj = md.load_dcd(dcd_files[i], psf_files[i], stride=stride)
        print('How many atoms?    %s' % traj.n_atoms)
        # align states onto first frame******
        traj.superpose(traj, frame=0)
        print(traj)
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
    mol : molecule with added conformers
    """
    for i, conf in enumerate(confs):
        if conf == None or conf.GetNumAtoms() == 0:
            continue # skip empty
    #     change mol name
    #     molname = sub('_traj',  str(i), mol.GetProp('_Name').lower()) # Replace pattern _traj -> conf #
    #     print(molname)

        # get conformer in Molecule object (*if rerunning, reload the conformers again because the IDs have been changed)
        c = conf.GetConformer(id=0)
        c.SetId(i) # check the Id of the original conformer that came with the Mol
        cid = c.GetId()
    #     # add each conformer to original input molecule
        mol.AddConformer(c, assignId=False)
    return mol


class ConformerGeneratorCustom(conformers.ConformerGenerator):
  # pruneRmsThresh=-1 means no pruning done here
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
      pbar.set_description("Processing %s" % i)
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
  rmsd = np.zeros((mol.GetNumConformers(), mol.GetNumConformers()),
                  dtype=float)
  pbar = tqdm(total=mol.GetNumConformers())
  # pct_prog = 100 / mol.GetNumConformers()
  for i, ref_conf in enumerate(mol.GetConformers()):
      pbar.set_description("Calculating RMSDs of conformer %s" % i)
      for j, fit_conf in enumerate(mol.GetConformers()):
          if i >= j:
              continue
#           rmsd[i, j] = AllChem.GetBestRMS(mol, mol, ref_conf.GetId(),
#                                       fit_conf.GetId())
          rmsd[i, j] = AllChem.GetConformerRMS(mol, ref_conf.GetId(),
                                                fit_conf.GetId())
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

