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

def test_default_NCI1():
    coordinator = Coordinator('NCI1', (0.9, 0.9, 2.3, 2.3, 'dirac'), './data/NCI1/data/')

    assert len(coordinator.graphs) == 4110


def test_default_proteins_tu():
    coordinator = Coordinator('proteins_tu', (1., 1., 1., 1., 'dirac'), './data/PROTEINS/data/')

    assert len(coordinator.graphs) == 1113


def test_default_enzymes():
    coordinator = Coordinator('enzymes', (1., 1., 1., 1., 'dirac'), './data/ENZYMES/data/')

    assert len(coordinator.graphs) == 600


def test_default_collab():
    coordinator = Coordinator('collab', (1., 1., 1., 1., 'dirac'), './data/COLLAB/data/')

    assert len(coordinator.graphs) == 5000


def test_default_reddit_binary():
    coordinator = Coordinator('reddit_binary', (1., 1., 1., 1., 'dirac'), './data/REDDIT-BINARY/data/')

    assert len(coordinator.graphs) == 2000


@pytest.mark.parametrize('dataset, folder_dataset, expected_msg',
                         [('aids', './data/AIDS/data/', 'The dataset aids is not available!'),
                          ('letter_lower', './data/Letter/Letter/LOW/', 'The dataset letter_lower is not available!')])
def test_error_dataset(dataset, folder_dataset, expected_msg):
    with pytest.raises(AssertionError) as info:
        coordinator = Coordinator(dataset, (0.4, 0.4, 0.4, 0.4, 'euclidean'), folder_dataset)

    error_msg = info.value.args[0]
    assert error_msg == expected_msg



