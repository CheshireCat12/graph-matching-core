import yaml


def load_config(experiment_name):
    """
    Load the configuration of the given experiment

    :param experiment_name:
    :return: dict - parameters of the experiments
    """
    with open(f'./configuration/configuration_{experiment_name}.yml') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)

    return parameters