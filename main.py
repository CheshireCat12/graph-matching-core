from graph_pkg.loader.loader_letter import LoaderLetter
from graph_pkg.loader.loader_mutagenicity import LoaderMutagenicity
from graph_pkg.loader.loader_AIDS import LoaderAIDS


def test_loader():
    loader = LoaderLetter()
    # loader = LoaderMutagenicity()
    # loader = LoaderAIDS()
    graphs = loader.load()
    print(graphs[0])
    print(graphs[1])


if __name__ == '__main__':
    test_loader()
