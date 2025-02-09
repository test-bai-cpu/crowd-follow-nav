import yaml


def load_config(config_path):
    """ Loading config file. """

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
