import argparse
import inspect
import os
import yaml
import logging

from pippin.config import get_config, mkdirs, get_logger
from pippin.manager import Manager

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="the name of the yml config file to run. For example: configs/default.yml")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO

    # Get base filename
    config_filename = os.path.basename(args.config).split(".")[0].upper()
    logging_folder = os.path.abspath(f"{get_config()['OUTPUT']['output_dir']}/{config_filename}")
    mkdirs(logging_folder)
    logging_filename = f"{logging_folder}/{config_filename}.log"

    # Initialise logging
    logging.basicConfig(
        level=level,
        format="[%(levelname)8s |%(filename)20s:%(lineno)3d |%(funcName)25s]   %(message)s",
        handlers=[
            logging.FileHandler(logging_filename),
            logging.StreamHandler()
        ]
    )

    logger = get_logger()
    logger.info(f"Logging streaming out, also saving to {logging_filename}")

    # Load YAML config file
    config_path = os.path.dirname(inspect.stack()[0][1]) + args.config
    assert os.path.exists(config_path), f"File {config_path} cannot be found."
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    manager = Manager(config_filename, config)
    manager.execute()

