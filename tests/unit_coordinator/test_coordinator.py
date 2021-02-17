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



