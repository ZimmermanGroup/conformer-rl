import conformer_rl
from conformer_rl.logging.train_logger import TrainLogger

def test_basic(mocker):
    mocker.patch('conformer_rl.logging.train_logger.mkdir')
    swriter = mocker.patch('conformer_rl.logging.train_logger.SummaryWriter')
    prt = mocker.patch('conformer_rl.logging.train_logger.print')

    writer = mocker.Mock()

    swriter.return_value = writer
    logger = TrainLogger(tag = "tag")

    assert logger.cache == {}
    swriter.assert_called_with(log_dir='data/tensorboard_log/tag/')

    logger.add_scalar('key', 100., 15)

    writer.add_scalar.assert_called_with('key', 100., 15, None)
    assert logger.cache['key'] == [[100.], [15]]
    prt.assert_called_once()

    logger.add_scalar('key', 120., 20)
    assert logger.cache['key'] == [[100., 120.], [15, 20]]



