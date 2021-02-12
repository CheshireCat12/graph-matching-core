import pytest

from graph_pkg.loader.loader_letter import LoaderLetter
# from graph_pkg.loader.loader_AIDS import LoaderAIDS
# from graph_pkg.loader.loader_mutagenicity import LoaderMutagenicity

@pytest.fixture()
def graphs_letter():
    loader_letter = LoaderLetter('./data/Letter/Letter/LOW/')
    graphs = loader_letter.load()

    return graphs

def test_loader_ordered_letter(graphs_letter):
    graph_0 = graphs_letter[0]
    graph_last = graphs_letter[-1]

    assert graph_0.name == 'AP1_0000'
    assert len(graph_0) == 6
    assert graph_last.name == 'ZP1_0149'
    assert len(graph_last) == 4
    assert len(graphs_letter) == 2250

# def test_loader_AIDS():
#     loader_AIDS = LoaderAIDS()
#     graphs = loader_AIDS.load()
#
#     assert len(graphs) == 2000
#
# def test_loader_mutagenicity():
#     loader_mutagenicity = LoaderMutagenicity()
#     graphs = loader_mutagenicity.load()
#
#     assert len(graphs) == 4337
