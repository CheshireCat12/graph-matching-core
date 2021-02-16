from experiments.run_matrix_distances import run_letter, run_AIDS, run_mutagenicity


def test_loader():
    run_letter()
    run_AIDS()
    run_mutagenicity()
    pass

if __name__ == '__main__':
    test_loader()
