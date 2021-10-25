from argparse import ArgumentParser

from bunch import Bunch

from experiments.run_bagging_knn import run_bagging_knn
from experiments.run_coarse_to_fine import run_coarse_to_fine
from experiments.run_complete_ged import run_complete_ged
from experiments.run_draw import run_draw
from experiments.run_h_knn import run_h_knn
from experiments.run_hierarchical import run_hierarchical
from experiments.run_knn import run_knn
from experiments.run_knn_lc import run_knn_lc
from experiments_gnn_embedding.run_knn_gnn_embedding import run_knn_gnn_embedding
from graph_pkg.utils.functions.load_config import load_config

__EXPERIMENTS = {
    'complete_ged': run_complete_ged,
    'knn': run_knn,
    'draw': run_draw,
    'hierarchical': run_hierarchical,
    'h_knn': run_h_knn,
    'knn_lc': run_knn_lc,
    'coarse_to_fine': run_coarse_to_fine,
    'bagging_knn': run_bagging_knn,
}

__EXPERIMENTS_GNN = {
    'knn': run_knn_gnn_embedding
}

__DATASETS = [
    'AIDS',
    'mutagenicity',
    'NCI1',
    'proteins_tu',
    'enzymes',
    'IMDB_binary'
]


def print_fancy_title(text, size_max=50):
    """
    Print the title in a fancy manner :)

    :param text:
    :param size_max:
    :return:
    """
    border = (size_max - len(text) - 4) // 2
    is_odd = len(text) % 2 != 0
    print(f'\n{"=" * size_max}\n'
          f'=={" " * border}{text}{" " * (border + is_odd)}==\n'
          f'{"=" * size_max}')


def run_experiment(args):
    parameters = load_config(args.exp, args.gnn)
    if args.all:
        for dataset in __DATASETS:
            args.dataset = dataset

            _run(args, parameters)
    else:
        _run(args, parameters)

    print_fancy_title('Final')


def _run(args, parameters):
    # Fusion the selected dataset parameters with the general parameters
    parameters = Bunch({**parameters[args.dataset], **parameters['general']})

    print_fancy_title('Parameters')
    print(parameters)

    print_fancy_title('Run')
    if args.gnn:
        __EXPERIMENTS_GNN[args.exp](parameters)
    else:
        __EXPERIMENTS[args.exp](parameters)


if __name__ == '__main__':
    parser = ArgumentParser(description='Run Experiments')
    parser.add_argument('-e', '--exp', type=str, required=True,
                        choices=__EXPERIMENTS.keys(),
                        help='Choose the experiment to run.')
    parser.add_argument('-d', '--dataset', type=str,
                        default='letter',
                        choices=['letter', 'AIDS', 'mutagenicity', 'NCI1',
                                 'proteins_tu', 'enzymes',
                                 'collab', 'reddit_binary', 'IMDB_binary'],
                        help='Choose the dataset.')
    parser.add_argument('-a', '--all', type=bool,
                        default=False,
                        choices=[True, False],
                        help='Run on all available datasets.')
    parser.add_argument('-g', '--gnn', type=bool,
                        default=False,
                        choices=[True, False],
                        help='Run the experiments with the GNN reduced graphs.')
    args = parser.parse_args()
    run_experiment(args)
