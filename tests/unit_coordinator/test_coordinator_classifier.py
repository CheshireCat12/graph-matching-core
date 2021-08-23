import pytest

from graph_pkg.utils.coordinator.coordinator_classifier import CoordinatorClassifier
from graph_pkg.utils.constants import get_default_lbls_to_code


###### Train #########

@pytest.mark.parametrize('dataset, folder_dataset, cost, expected_size',
                         [
                          # ('letter', './data/Letter/Letter/HIGH/', 'euclidean', 750),
                          # ('AIDS', './data/AIDS/data/', 'dirac', 250),
                          # ('mutagenicity', './data/Mutagenicity/data/', 'dirac',  1500),
                          ('NCI1', './data/NCI1/data/', 'dirac', 1500),
                          # ('proteins_tu', './data/PROTEINS/data/', 'dirac', 660),
                          # ('enzymes', './data/ENZYMES/data/', 'dirac', 360),
                          # ('collab', './data/COLLAB/data/', 'dirac', 3000),
                          # ('reddit_binary', './data/REDDIT-BINARY/data/', 'dirac', 1200),
                          ])
def test_train_split(dataset, folder_dataset, cost, expected_size):
    coordinator = CoordinatorClassifier(dataset, (0.9, 0.9, 2.3, 2.3, cost), folder_dataset)

    X_train, y_train = coordinator.train_split()

    assert len(X_train) == expected_size
    assert len(y_train) == expected_size

    loader_split = coordinator.loader_split
    data = loader_split.load_train_split()
    graph_to_lbl = {graph_filename: lbl for graph_filename, lbl in data}

    for idx, (graph, lbl) in enumerate(zip(X_train, y_train)):
        expected_lbl = graph_to_lbl[graph.filename]

        if idx == 0:
            print(graph.filename)
            assert False
        assert lbl == expected_lbl


####### test ###########

@pytest.mark.parametrize('dataset, folder_dataset, cost, expected_size',
                         [
                             # ('letter', './data/Letter/Letter/HIGH/', 'euclidean', 750),
                             # ('AIDS', './data/AIDS/data/', 'dirac', 1500),
                             # ('mutagenicity', './data/Mutagenicity/data/', 'dirac',  2337),
                             ('NCI1', './data/NCI1/data/', 'dirac', 2110),
                             # ('proteins_tu', './data/PROTEINS/data/', 'dirac', 233),
                             # ('enzymes', './data/ENZYMES/data/', 'dirac', 120),
                             # ('collab', './data/COLLAB/data/', 'dirac', 1000),
                             # ('reddit_binary', './data/REDDIT-BINARY/data/', 'dirac', 400),
                         ])
def test_test_split(dataset, folder_dataset, cost, expected_size):
    coordinator = CoordinatorClassifier(dataset, (0.9, 0.9, 2.3, 2.3, cost), folder_dataset)

    X_test, y_test = coordinator.test_split()

    assert len(X_test) == expected_size
    assert len(y_test) == expected_size

    loader_split = coordinator.loader_split
    data = loader_split.load_test_split()
    graph_to_lbl = {graph_filename: lbl for graph_filename, lbl in data}

    for idx, (graph, lbl) in enumerate(zip(X_test, y_test)):
        expected_lbl = graph_to_lbl[graph.filename]

        if idx == 127:
            print(graph.filename)
            assert False

        assert lbl == expected_lbl


############## validation ##################

@pytest.mark.parametrize('dataset, folder_dataset, cost, expected_size',
                         [
                             ('letter', './data/Letter/Letter/HIGH/', 'euclidean', 750),
                             ('AIDS', './data/AIDS/data/', 'dirac', 250),
                             ('mutagenicity', './data/Mutagenicity/data/', 'dirac',  500),
                             ('NCI1', './data/NCI1/data/', 'dirac', 500),
                             ('proteins_tu', './data/PROTEINS/data/', 'dirac', 220),
                             ('enzymes', './data/ENZYMES/data/', 'dirac', 120),
                             ('collab', './data/COLLAB/data/', 'dirac', 1000),
                             ('reddit_binary', './data/REDDIT-BINARY/data/', 'dirac', 400),
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


############## code labels ##################

@pytest.mark.parametrize('dataset, folder_dataset, cost, expected_size',
                         [
                             ('letter', './data/Letter/Letter/HIGH/', 'euclidean', 750),
                             ('AIDS', './data/AIDS/data/', 'dirac', 250),
                             ('mutagenicity', './data/Mutagenicity/data/', 'dirac',  1500),
                             ('proteins_tu', './data/PROTEINS/data/', 'dirac', 660),
                             ('enzymes', './data/ENZYMES/data/', 'dirac', 360),
                             ('collab', './data/COLLAB/data/', 'dirac', 3000),
                             ('reddit_binary', './data/REDDIT-BINARY/data/', 'dirac', 1200),
                         ])
def test_train_with_encoded_lbls(dataset, folder_dataset, cost, expected_size):
    coordinator = CoordinatorClassifier(dataset, (0.9, 0.9, 2.3, 2.3, cost), folder_dataset)

    X_train, y_train = coordinator.train_split(conv_lbl_to_code=True)

    assert len(X_train) == expected_size
    assert len(y_train) == expected_size

    default_lbls_to_code = get_default_lbls_to_code()
    lbls_to_code = default_lbls_to_code[dataset]

    loader_split = coordinator.loader_split
    data = loader_split.load_train_split()
    graph_to_lbl = {graph_filename: lbl for graph_filename, lbl in data}

    for graph, lbl in zip(X_train, y_train):
        expected_lbl = graph_to_lbl[graph.filename]
        expected_lbl = lbls_to_code[expected_lbl]
        assert lbl == expected_lbl
