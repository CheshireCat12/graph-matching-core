import pytest

from graph_pkg.loader.loader_train_test_val_split import LoaderTrainTestValSplit

@pytest.mark.parametrize('folder_dataset, expected_size, first_name_expected, first_lbl_expected, last_name_expected, last_lbl_expected',
                         [('./data/Letter/Letter/HIGH/', 750, 'AP1_0000.gxl', 'A', 'ZP1_0049.gxl', 'Z'),
                          ('./data/AIDS/data/', 250, '11808.gxl', 'a', '27997.gxl', 'i'),
                          ('./data/Mutagenicity/data/', 1500, 'molecule_2767.gxl', 'mutagen', 'molecule_4337.gxl', 'nonmutagen')
                          ])
def test_split_train(folder_dataset, expected_size, first_name_expected, first_lbl_expected, last_name_expected, last_lbl_expected):
    loader = LoaderTrainTestValSplit(folder_dataset)
    data = loader.load_train_split()
    first_graph, first_lbl = data[0]
    last_graph, last_lbl = data[-1]
    assert len(data) == expected_size
    assert first_graph == first_name_expected
    assert first_lbl == first_lbl_expected
    assert last_graph == last_name_expected
    assert last_lbl == last_lbl_expected


@pytest.mark.parametrize('folder_dataset, expected_size',
                         [('./data/Letter/Letter/HIGH/', 750),
                          ('./data/AIDS/data/', 250),
                          ('./data/Mutagenicity/data/', 500)
                          ])
def test_split_val(folder_dataset, expected_size):
    loader = LoaderTrainTestValSplit(folder_dataset)
    data = loader.load_val_split()
    assert len(data) == expected_size


@pytest.mark.parametrize('folder_dataset, expected_size',
                         [('./data/Letter/Letter/HIGH/', 750),
                          ('./data/AIDS/data/', 1500),
                          ('./data/Mutagenicity/data/', 2337)
                          ])
def test_split_test(folder_dataset, expected_size):
    loader = LoaderTrainTestValSplit(folder_dataset)
    data = loader.load_test_split()
    assert len(data) == expected_size
