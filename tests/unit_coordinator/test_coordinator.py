import pytest
from graph_pkg.coordinator import Coordinator

def test_default_letter():
    coordinator = Coordinator('letter_high', (0.9, 0.9, 2.3, 2.3, 'euclidean'))

    assert len(coordinator.graphs) == 2250

def test_default_AIDS():
    coordinator = Coordinator('AIDS', (0.9, 0.9, 2.3, 2.3, 'dirac'))

    assert len(coordinator.graphs) == 2000

def test_default_mutagenicity():
    coordinator = Coordinator('mutagenicity', (0.9, 0.9, 2.3, 2.3, 'dirac'))

    assert len(coordinator.graphs) == 4337

@pytest.mark.parametrize('dataset, expected_msg',
                         [('aids', 'The dataset aids is not available!'),
                          ('letter_lower', 'The dataset letter_lower is not available!')])
def test_error_dataset(dataset, expected_msg):
    with pytest.raises(ValueError) as info:
        coordinator = Coordinator(dataset, (0.4, 0.4, 0.4, 0.4, 'euclidean'))

    error_msg = info.value.args[0]
    assert error_msg == expected_msg