from conformer_rl.utils import chem_utils
import numpy as np

def test_tfd_matrix(mocker):
    tf = mocker.patch('conformer_rl.utils.chem_utils.TorsionFingerprints')
    tf.GetTFDMatrix.return_value = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    mat = chem_utils.tfd_matrix('mol')
    assert np.array_equal(mat, np.array(
        [[0., 3, 5, 9, 15],
        [3, 0, 7, 11, 17,],
        [5, 7, 0, 13, 19],
        [9, 11, 13, 0, 21],
        [15, 17, 19, 21, 0]]
    ))