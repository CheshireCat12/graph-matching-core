from argparse import ArgumentParser
from bunch import Bunch

from graph_pkg.utils.functions.load_config import load_config
from experiments.run_complete_ged import run_complete_ged
from experiments.run_knn import run_knn
from experiments.run_draw import run_draw
from experiments.run_hierarchical import run_hierarchical
from experiments.run_h_knn import run_h_knn

__EXPERIMENTS = {'complete_ged': run_complete_ged,
                 'knn': run_knn,
                 'draw': run_draw,
                 'hierarchical': run_hierarchical,
                 'h_knn': run_h_knn}

def print_fancy_title(text, size_max=50):
    """
    Print the title in a fancy manner :)

    :param text:
    :param size_max:
    :return:
    """
    border = (size_max - len(text) - 4) // 2
    is_odd = len(text) % 2 != 0
    print(f'\n{"="*size_max}\n'
          f'=={" "*border}{text}{" "*(border+is_odd)}==\n'
          f'{"="*size_max}')


def run_experiment(args):
    parameters = load_config(args.exp)

    # Fusion the selected dataset parameters with the general parameters
    parameters = Bunch({**parameters[args.dataset], **parameters['general']})

    print_fancy_title('Parameters')
    print(parameters)

    print_fancy_title('Run')
    __EXPERIMENTS[args.exp](parameters)

    # if args.exp == 'complete_ged':
    #     run_complete_ged(parameters)
    # elif args.exp == 'knn':
    #     run_knn(parameters)
    # elif args.exp == 'draw':
    #     run_draw(parameters)


if __name__ == '__main__':
    parser = ArgumentParser(description='Run Experiments')
    parser.add_argument('-e', '--exp', type=str, required=True,
                        choices=__EXPERIMENTS.keys(), # ['complete_ged', 'knn', 'draw'],
                        help='Choose the experiment to run.')
    parser.add_argument('-d', '--dataset', type=str,
                        default='letter',
                        choices=['letter', 'AIDS', 'mutagenicity', 'NCI1'],
                        help='Choose the dataset.')
    args = parser.parse_args()
    run_experiment(args)
