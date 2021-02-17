import pytest

from graph_pkg.utils.coordinator.coordinator_classifier import CoordinatorClassifier

###### Train #########

@pytest.mark.parametrize('dataset, folder_dataset, cost, expected_size',
                         [
                          ('letter', './data/Letter/Letter/HIGH/', 'euclidean', 750),
                          ('AIDS', './data/AIDS/data/', 'dirac', 250),
                          ('mutagenicity', './data/Mutagenicity/data/', 'dirac',  1500)
                          ])
def test_train_split(dataset, folder_dataset, cost, expected_size):
    coordinator = CoordinatorClassifier(dataset, (0.9, 0.9, 2.3, 2.3, cost), folder_dataset)

    X_train, y_train = coordinator.train_split()

    assert len(X_train) == expected_size
    assert len(y_train) == expected_size

    loader_split = coordinator.loader_split
    data = loader_split.load_train_split()
    graph_to_lbl = {graph_filename: lbl for graph_filename, lbl in data}

    for graph, lbl in zip(X_train, y_train):
        expected_lbl = graph_to_lbl[graph.filename]
        assert lbl == expected_lbl


####### test ###########

@pytest.mark.parametrize('dataset, folder_dataset, cost, expected_size',
                         [
                             ('letter', './data/Letter/Letter/HIGH/', 'euclidean', 750),
                             ('AIDS', './data/AIDS/data/', 'dirac', 1500),
                             ('mutagenicity', './data/Mutagenicity/data/', 'dirac',  2337)
                         ])
def test_test_split(dataset, folder_dataset, cost, expected_size):
    coordinator = CoordinatorClassifier(dataset, (0.9, 0.9, 2.3, 2.3, cost), folder_dataset)

    X_test, y_test = coordinator.test_split()

    assert len(X_test) == expected_size
    assert len(y_test) == expected_size

    loader_split = coordinator.loader_split
    data = loader_split.load_test_split()
    graph_to_lbl = {graph_filename: lbl for graph_filename, lbl in data}

    for graph, lbl in zip(X_test, y_test):
        expected_lbl = graph_to_lbl[graph.filename]
        assert lbl == expected_lbl


############## validation ##################

@pytest.mark.parametrize('dataset, folder_dataset, cost, expected_size',
                         [
                             ('letter', './data/Letter/Letter/HIGH/', 'euclidean', 750),
                             ('AIDS', './data/AIDS/data/', 'dirac', 250),
                             ('mutagenicity', './data/Mutagenicity/data/', 'dirac',  500)
                         ])
def test_val_split(dataset, folder_dataset, cost, expected_size):
    coordinator = CoordinatorClassifier(dataset, (0.9, 0.9, 2.3, 2.3, cost), folder_dataset)

    X_val, y_val = coordinator.val_split()

    assert len(X_val) == expected_size
    assert len(y_val) == expected_size

    loader_split = coordinator.loader_split
    data = loader_split.load_val_split()
    graph_to_lbl = {graph_filename: lbl for graph_filename, lbl in data}

    for graph, lbl in zip(X_val, y_val):
        expected_lbl = graph_to_lbl[graph.filename]
        assert lbl == expected_lbl
