import configparser
import inspect
import os
import logging
import hashlib
import shutil


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
    filename = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/../cfg.ini")
    config = configparser.ConfigParser()
    config.read(filename)
    return config


def get_hash(input_string):
    return hashlib.sha256(input_string.encode('utf-8')).hexdigest()


def get_logger():
    return logging.getLogger("pippin")


def chown_dir(directory):
    global_config = get_config()
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            shutil.chown(d, group=global_config["SNANA"]["group"])
        for f in files:
            shutil.chown(f, group=global_config["SNANA"]["group"])


if __name__ == "__main__":
    c = get_config()
    print(c.sections())
    print(c.get("SNANA", "sim_dir"))
    print(c["OUTPUT"].getint("ping_frequency"))
