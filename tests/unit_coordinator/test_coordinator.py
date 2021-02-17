import pytest

from graph_pkg.utils.coordinator.coordinator import Coordinator

def test_default_letter():
    coordinator = Coordinator('letter', (0.9, 0.9, 2.3, 2.3, 'euclidean'), './data/Letter/Letter/HIGH/')

    assert len(coordinator.graphs) == 2250

def test_default_AIDS():
    coordinator = Coordinator('AIDS', (0.9, 0.9, 2.3, 2.3, 'dirac'), './data/AIDS/data/')

    assert len(coordinator.graphs) == 2000

def test_default_mutagenicity():
    coordinator = Coordinator('mutagenicity', (0.9, 0.9, 2.3, 2.3, 'dirac'), './data/Mutagenicity/data/')

    assert len(coordinator.graphs) == 4337

@pytest.mark.parametrize('dataset, folder_dataset, expected_msg',
                         [('aids', './data/AIDS/data/', 'The dataset aids is not available!'),
                          ('letter_lower', './data/Letter/Letter/LOW/', 'The dataset letter_lower is not available!')])
def test_error_dataset(dataset, folder_dataset, expected_msg):
    with pytest.raises(AssertionError) as info:
        coordinator = Coordinator(dataset, (0.4, 0.4, 0.4, 0.4, 'euclidean'), folder_dataset)

    error_msg = info.value.args[0]
    assert error_msg == expected_msg


# @pytest.mark.parametrize('dataset, folder_dataset, cost, expected_size',
#                          [('letter_high', './data/Letter/Letter/HIGH/', 'euclidean', 750),
#                           ('AIDS', './data/AIDS/data/', 'dirac', 250),
#                           ('mutagenicity', './data/Mutagenicity/data/', 'dirac',  1500)
#                           ])
# def test_train_split(dataset, folder_dataset, cost, expected_size):
#     coordinator = Coordinator(dataset, (0.9, 0.9, 2.3, 2.3, cost), folder_dataset)
#
#     X_train, y_train = coordinator.train_split()
#
#
#     assert len(X_train) == expected_size
#     assert len(y_train) == expected_size
