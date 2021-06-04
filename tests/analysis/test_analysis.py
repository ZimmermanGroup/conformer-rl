from conformer_rl.analysis import analysis

def test_load_pickle(mocker):
    open = mocker.patch('conformer_rl.analysis.analysis.open')
    load = mocker.patch('conformer_rl.analysis.analysis.pickle.load')
    file = mocker.Mock()


    open.return_value = file

    analysis._load_from_pickle('filename')

    open.assert_called_with('filename', 'rb')
    load.assert_called_with(file)

def test_load_data_single(mocker):
    def load_pickle(path):
        d = {
            'path1': {
                'data1': 1
            }
        }
        return d[path]
    pickle = mocker.patch('conformer_rl.analysis.analysis._load_from_pickle')
    pickle.side_effect = load_pickle

    data = analysis.load_data_from_pickle(paths = ['path1'])
    assert data == {'indices': ['test0'], 'data1': [1]}

def test_load_data_multiple(mocker):
    def load_pickle(path):
        d = {
            'data1': {
            'total_rewards': 'data1_total_rewards',
            'mol': 'data1_molecule',
            'rewards': ['data1_step1_rewards', 'data1_step2_rewards', 'data1_step3_rewards', 'data1_step4_rewards']
            },
            'data2': {
            'total_rewards': 'data2_total_rewards',
            'mol': 'data2_molecule',
            'rewards': ['data2_step1_rewards', 'data2_step2_rewards', 'data2_step3_rewards', 'data2_step4_rewards']
            },
            'data3': {
            'total_rewards': 'data3_total_rewards',
            'mol': 'data3_molecule',
            'rewards': ['data3_step1_rewards', 'data3_step2_rewards', 'data3_step3_rewards', 'data3_step4_rewards']
            },
        }

        return d[path]

    pickle = mocker.patch('conformer_rl.analysis.analysis._load_from_pickle')
    pickle.side_effect = load_pickle
    data = analysis.load_data_from_pickle(paths = ['data1', 'data2', 'data3'], indices = ['PPO', 'PPO_recurrent', 'A2C'])
    assert data == {
        'indices': ['PPO', 'PPO_recurrent', 'A2C'],
        'total_rewards': [
            'data1_total_rewards',
            'data2_total_rewards',
            'data3_total_rewards'
        ],
        'mol': [
            'data1_molecule',
            'data2_molecule',
            'data3_molecule'
        ],
        'rewards': [
            ['data1_step1_rewards', 'data1_step2_rewards', 'data1_step3_rewards', 'data1_step4_rewards'],
            ['data2_step1_rewards', 'data2_step2_rewards', 'data2_step3_rewards', 'data2_step4_rewards'],
            ['data3_step1_rewards', 'data3_step2_rewards', 'data3_step3_rewards', 'data3_step4_rewards']
        ]
    }

    
    

def test_list_keys():
    d = {'test1': 1, 'test2': 2, 'test3': 3}
    assert analysis.list_keys(d) == ['test1', 'test2', 'test3']
