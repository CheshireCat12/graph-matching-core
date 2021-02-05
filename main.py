from graph_pkg.loader.loader_letter import LoaderLetter
from graph_pkg.loader.loader_mutagenicity import LoaderMutagenicity
from graph_pkg.loader.loader_AIDS import LoaderAIDS


def test_loader():
    # loader = LoaderLetter('./data/Letter/Letter/LOW/')
    loader = LoaderMutagenicity('./data/Mutagenicity/data/')
    # loader = LoaderAIDS('./data/AIDS/data/')
    graphs = loader.load()


if __name__ == '__main__':
    test_loader()
