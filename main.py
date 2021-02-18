from argparse import ArgumentParser

from graph_pkg.utils.functions.load_config import load_config
from experiments.run_complete_ged import run_complete_ged
from experiments.run_knn import run_knn

def run_experiment(args):
    parameters = load_config(args.exp)

    if args.exp == 'complete_ged':
        run_complete_ged(parameters[args.dataset])
    elif args.exp == 'knn':
        run_knn(parameters[args.dataset])


if __name__ == '__main__':
    parser = ArgumentParser(description='Run Experiments')
    parser.add_argument('-e', '--exp', type=str, required=True,
                        choices=['complete_ged', 'knn'],
                        help='Choose the experiment to run.')
    parser.add_argument('-d', '--dataset', type=str,
                        default='letter',
                        choices=['letter', 'AIDS', 'mutagenicity'],
                        help='Choose the dataset.')
    args = parser.parse_args()
    run_experiment(args)
