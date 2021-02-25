from argparse import ArgumentParser
from collections import namedtuple
from bunch import Bunch

from graph_pkg.utils.functions.load_config import load_config
from experiments.run_complete_ged import run_complete_ged
from experiments.run_knn import run_knn
from experiments.run_draw import run_draw

def run_experiment(args):
    parameters = load_config(args.exp)

    # Fusion the selected dataset parameters with the general parameters
    parameters = Bunch({**parameters[args.dataset], **parameters['general']})

    if args.exp == 'complete_ged':
        run_complete_ged(parameters)
    elif args.exp == 'knn':
        run_knn(parameters)
    elif args.exp == 'draw':
        run_draw(parameters)


if __name__ == '__main__':
    parser = ArgumentParser(description='Run Experiments')
    parser.add_argument('-e', '--exp', type=str, required=True,
                        choices=['complete_ged', 'knn', 'draw'],
                        help='Choose the experiment to run.')
    parser.add_argument('-d', '--dataset', type=str,
                        default='letter',
                        choices=['letter', 'AIDS', 'mutagenicity'],
                        help='Choose the dataset.')
    args = parser.parse_args()
    run_experiment(args)
