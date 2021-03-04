import pytest

from graph_pkg.loader.loader_AIDS import LoaderAIDS
from graph_pkg.loader.loader_letter import LoaderLetter
from graph_pkg.loader.loader_mutagenicity import LoaderMutagenicity
from graph_pkg.loader.loader_NCI1 import LoaderNCI1


############## LETTER ##############

@pytest.fixture()
def graphs_letter_low():
    loader_letter = LoaderLetter('./data/Letter/Letter/LOW')
    graphs = loader_letter.load()

    return graphs

@pytest.mark.parametrize('folder, num_graphs',
                         [('./data/Letter/Letter/LOW/', 2250),
                          ('./data/Letter/Letter/MED', 2250),
                          ('./data/Letter/Letter/HIGH', 2250)])
def test_all_letters(folder, num_graphs):
    loader_letter = LoaderLetter(folder)
    graphs = loader_letter.load()

    assert len(graphs) == num_graphs

@pytest.mark.parametrize('idx_graph, gr_name, num_nodes',
                         [(0, 'AP1_0000', 6),
                          (-1, 'ZP1_0149', 4)
                          ])
def test_loader_ordered_letter_low(graphs_letter_low, idx_graph, gr_name, num_nodes):
    graph = graphs_letter_low[idx_graph]

    assert graph.name == gr_name
    assert len(graph) == num_nodes

####################### AIDS ###########################

@pytest.fixture()
def graphs_AIDS():
    loader_AIDS = LoaderAIDS()
    graphs = loader_AIDS.load()

    return graphs

def test_loader_AIDS_size(graphs_AIDS):
    assert len(graphs_AIDS) == 2000


@pytest.mark.parametrize('idx_graph, gr_name, num_nodes',
                         [(0, 'molid2184', 10),
                          (-1, 'molid405554', 11),
                          ])
def test_loader_AIDS_value(graphs_AIDS, idx_graph, gr_name, num_nodes):
    graph = graphs_AIDS[idx_graph]

    assert graph.name == gr_name
    assert len(graph) == num_nodes


######################### Mutagenicity #######################

@pytest.fixture()
def graphs_mutagenicity():
    loader = LoaderMutagenicity()
    graphs = loader.load()

    return graphs


def test_loader_mutagenicity_len(graphs_mutagenicity):
    assert len(graphs_mutagenicity) == 4337


@pytest.mark.parametrize('idx_graph, gr_name, num_nodes',
                         [(0, 'molecule_1', 43),
                          (-1, 'molecule_999', 32),
                          ])
def test_loader_mutagenicity_value(graphs_mutagenicity, idx_graph, gr_name, num_nodes):
    graph = graphs_mutagenicity[idx_graph]

    assert graph.name == gr_name
    assert len(graph) == num_nodes

######################### NCI1 #######################

@pytest.fixture()
def graphs_NCI1():
    loader = LoaderNCI1()
    graphs = loader.load()

    return graphs


def test_loader_NCI1_len(graphs_NCI1):
    assert len(graphs_NCI1) == 4110


@pytest.mark.parametrize('idx_graph, gr_name, num_nodes',
                         [(0, 'molecule_0', 21),
                          (-1, 'molecule_999', 26),
                          ])
def test_loader_NCI1_value(graphs_NCI1, idx_graph, gr_name, num_nodes):
    graph = graphs_NCI1[idx_graph]

    assert graph.name == gr_name
    assert len(graph) == num_nodes

#################### Loader with wrong folder ###############

def test_loader_with_wrong_folder():
    folder = './data/Letter/Letter'
    loader = LoaderLetter(folder)
    with pytest.raises(FileNotFoundError) as execinfo:
        loader.load()
    print(execinfo.value.args)

    msg_error = execinfo.value.args[0]
    msg_expected = f'No graphs found in {folder}'
    assert msg_error == msg_expected
