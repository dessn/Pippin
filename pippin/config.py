import configparser
import inspect
import os
import logging


def singleton(fn):
    instance = None

    def get(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = fn(*args, **kwargs)
        return instance
    return get


@singleton
def get_config():
    filename = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/../config.cfg")
    config = configparser.ConfigParser()
    config.read(filename)
    return config


def get_logger():
    return logging.getLogger("pippin")


if __name__ == "__main__":
    c = get_config()
    print(c.sections())
    print(c.get("SNANA", "sim_dir"))
