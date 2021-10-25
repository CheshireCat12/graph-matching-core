import yaml


def load_config(experiment_name, gnn_exp=False):
    """
    Load the configuration of the given experiment

    :param experiment_name:
    :return: dict - parameters of the experiments
    """
    gnn_ext = '_gnn_embedding' if gnn_exp else ''
    with open(f'./configuration{gnn_ext}/configuration{gnn_ext}_{experiment_name}.yml') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)

    return parameters