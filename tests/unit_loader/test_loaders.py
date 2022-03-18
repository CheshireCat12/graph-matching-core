import pytest

from graph_pkg_core.loader.loader_vector import LoaderVector


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

# ####################### AIDS ###########################
#
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

######################### Proteins TU #######################

@pytest.fixture()
def graphs_proteins_tu():
    loader = LoaderProteinsTU()
    graphs = loader.load()

    return graphs


def test_loader_proteins_tu_len(graphs_proteins_tu):
    assert len(graphs_proteins_tu) == 1113


@pytest.mark.parametrize('idx_graph, gr_name, num_nodes',
                         [(0, 'molecule_0', 42),
                          (-1, 'molecule_999', 15),
                          ])
def test_loader_proteins_tu_value(graphs_proteins_tu, idx_graph, gr_name, num_nodes):
    graph = graphs_proteins_tu[idx_graph]

    assert graph.name == gr_name
    assert len(graph) == num_nodes


######################### Enzymes #######################

@pytest.fixture()
def graphs_enzymes():
    loader = LoaderEnzymes()
    graphs = loader.load()

    return graphs


def test_loader_enzymes_len(graphs_enzymes):
    assert len(graphs_enzymes) == 600


@pytest.mark.parametrize('idx_graph, gr_name, num_nodes',
                         [(0, 'molecule_0', 37),
                          (-1, 'molecule_99', 5),
                          ])
def test_loader_enzymes_value(graphs_enzymes, idx_graph, gr_name, num_nodes):
    graph = graphs_enzymes[idx_graph]

    assert graph.name == gr_name
    assert len(graph) == num_nodes



######################### collab #######################

@pytest.fixture()
def graphs_collab():
    loader = LoaderCollab()
    graphs = loader.load()

    return graphs


def test_loader_collab_len(graphs_collab):
    assert len(graphs_collab) == 5000


@pytest.mark.parametrize('idx_graph, gr_name, num_nodes',
                         [(0, 'molecule_0', 45),
                          (-1, 'molecule_999', 60),
                          ])
def test_loader_collab_value(graphs_collab, idx_graph, gr_name, num_nodes):
    graph = graphs_collab[idx_graph]

    assert graph.name == gr_name
    assert len(graph) == num_nodes


######################### Reddit Binary #######################

@pytest.fixture()
def graphs_reddit_binary():
    loader = LoaderRedditBinary()
    graphs = loader.load()

    return graphs


def test_loader_reddit_binary_len(graphs_reddit_binary):
    assert len(graphs_reddit_binary) == 2000


@pytest.mark.parametrize('idx_graph, gr_name, num_nodes',
                         [(0, 'molecule_0', 218),
                          (-1, 'molecule_999', 373),
                          ])
def test_loader_reddit_binary_value(graphs_reddit_binary, idx_graph, gr_name, num_nodes):
    graph = graphs_reddit_binary[idx_graph]

    assert graph.name == gr_name
    assert len(graph) == num_nodes

######################### Protein #######################

# @pytest.fixture()
# def graphs_Protein():
#     loader = LoaderProtein()
#     graphs = loader.load()
#
#     return graphs
#
#
# def test_loader_Protein_len(graphs_Protein):
#     assert len(graphs_Protein) == 600
#
#
# @pytest.mark.parametrize('idx_graph, gr_name, num_nodes',
#                          [(0, 'pdb1h3e', 37),
#                           (1, 'pdb1a8h', 32),
#                           ])
# def test_loader_Protein_value(graphs_Protein, idx_graph, gr_name, num_nodes):
#     graph = graphs_Protein[idx_graph]
#
#     assert graph.name == gr_name
#     assert len(graph) == num_nodes
#
# #################### Loader with wrong folder ###############
#
# def test_loader_with_wrong_folder():
#     folder = './data/Letter/Letter'
#     loader = LoaderLetter(folder)
#     with pytest.raises(FileNotFoundError) as execinfo:
#         loader.load()
#     print(execinfo.value.args)
#
#     msg_error = execinfo.value.args[0]
#     msg_expected = f'No graphs found in {folder}'
#     assert msg_error == msg_expected
#