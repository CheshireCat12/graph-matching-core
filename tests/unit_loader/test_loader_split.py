import pytest

from graph_pkg.loader.loader_train_test_val_split import LoaderTrainTestValSplit

@pytest.mark.parametrize('folder_dataset, expected_size, first_name_expected, first_lbl_expected, last_name_expected, last_lbl_expected',
                         [('./data/Letter/Letter/HIGH/', 750, 'AP1_0000.gxl', 'A', 'ZP1_0049.gxl', 'Z'),
                          ('./data/AIDS/data/', 250, '11808.gxl', 'a', '27997.gxl', 'i'),
                          ('./data/Mutagenicity/data/', 1500, 'molecule_2767.gxl', 'mutagen', 'molecule_4337.gxl', 'nonmutagen')
                          ])
def test_split_train(folder_dataset, expected_size, first_name_expected, first_lbl_expected, last_name_expected, last_lbl_expected):
    loader = LoaderTrainTestValSplit(folder_dataset)
    X_train, y_train = loader.train_split()
    assert len(X_train) == expected_size
    assert len(y_train) == expected_size
    assert X_train[0] == first_name_expected
    assert y_train[0] == first_lbl_expected
    assert X_train[-1] == last_name_expected
    assert y_train[-1] == last_lbl_expected


@pytest.mark.parametrize('folder_dataset, expected_size',
                         [('./data/Letter/Letter/HIGH/', 750),
                          ('./data/AIDS/data/', 1500),
                          ('./data/Mutagenicity/data/', 2337)
                          ])
def test_split_test(folder_dataset, expected_size):
    loader = LoaderTrainTestValSplit(folder_dataset)
    X_train, y_train = loader.test_split()
    assert len(X_train) == expected_size
    assert len(y_train) == expected_size


@pytest.mark.parametrize('folder_dataset, expected_size',
                         [('./data/Letter/Letter/HIGH/', 750),
                          ('./data/AIDS/data/', 250),
                          ('./data/Mutagenicity/data/', 500)
                          ])
def test_split_val(folder_dataset, expected_size):
    loader = LoaderTrainTestValSplit(folder_dataset)
    X_train, y_train = loader.val_split()
    assert len(X_train) == expected_size
    assert len(y_train) == expected_size
