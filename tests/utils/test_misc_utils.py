from conformer_rl.utils import misc_utils
import datetime
import os
import torch
import numpy as np

def test_time(mocker):
    dt = mocker.patch('conformer_rl.utils.misc_utils.datetime')
    dt.now.return_value = datetime.datetime(2000, 5, 20, 14, 32, 33)
    assert misc_utils.current_time() == "20-05-2000_14:32:33"

def test_thread():
    misc_utils.set_one_thread()
    assert os.environ['OMP_NUM_THREADS'] == '1'
    assert os.environ['MKL_NUM_THREADS'] == '1'

def test_utils(mocker):
    path = mocker.patch('conformer_rl.utils.misc_utils.Path')
    file = mocker.Mock()
    path.return_value = file

    misc_utils.mkdir("filename")
    path.assert_called_with("filename")
    file.mkdir.assert_called_once()

    assert misc_utils.to_np(torch.tensor(5)) == np.array(5)

    model = mocker.Mock()
    model.state_dict.return_value = 'state_dict'
    t = mocker.patch('conformer_rl.utils.misc_utils.torch')
    misc_utils.save_model(model, "filename")
    t.save.assert_called_with('state_dict', 'filename')

    misc_utils.load_model(model, "filename")
    t.load.assert_called_with("filename")
    

