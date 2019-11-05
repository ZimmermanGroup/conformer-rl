from rdkit import Chem, RDConfig, rdBase
from rdkit import rdBase
from rdkit.Chem import AllChem, TorsionFingerprints
import numpy as np

def molfile_to_sdf(files, outfile):
    """
    Parameters
    ----------
    files: list of .mol files
    outfile: path of writing sdf file
    """
    with open(outfile , 'w') as sdf_file:
        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    sdf_file.write(line)
            sdf_file.write("$$$$\n")

def load_from_sdf(sdf_file):
    """
    Parameters
    ----------
    sdf_file: path to SDF file

    Returns
    -------
    sdf_mols: list of Molecules
    """
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False) #, strictParsing=False
    print ("Loading SDF file...", end='\t')
    sdf_mols = [mol for mol in suppl]
#     print(type(suppl))
#     print(type(sdf_mols))
    if len(sdf_mols) == 1:
        print("Molecule loaded.")
    else:
        print("{} conformers as Molecule type loaded".format(len(sdf_mols)))
    return sdf_mols

def check_symmetric(mat, rtol=1e-05, atol=1e-08):
    return np.allclose(mat, mat.T, rtol=rtol, atol=atol)

# matrix reshaping
def symmetric_to_array(mat):
    """
    converts symmetric nXn matrix to 1D array (good use for Butina clustering in rdkit)
    e.g. rmsmatrix = [a,
                      b, c,
                      d, e, f,
                      g, h, i, j]
    Parameters
    ----------
    mat: symmetric matrix

    Returns
    -------
    1D array of the lower triangle part of the symmetric matrix
    """
    n = mat.shape[1]
    ix_lt = np.tril_indices(n, k=-1)
    return mat[ix_lt]

def array_to_lower_triangle(arr):
    """
    Converts list to lower triangle mat with zeroes as other elements of the matrix
    """
    n = int(np.sqrt(len(arr)*2))+1
    idx = np.tril_indices(n, k=-1, m=n)
    lt_mat = np.zeros((n,n))
    lt_mat[idx] = arr
    # if get_symm == True:
    #     return lt_mat + np.transpose(lt_mat) # symmetric matrix
    return lt_mat
