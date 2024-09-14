import argparse
import os
import yaml
import logging
import coloredlogs
import signal
import sys
from pippin.config import (
    mkdirs,
    get_logger,
    get_output_dir,
    chown_file,
    get_config,
    chown_dir,
)
from pippin.manager import Manager
from colorama import init
import socket


class MessageStore(logging.Handler):
    store = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.store = {}

    def emit(self, record):
        l = record.levelname
        if l not in self.store:
            self.store[l] = []
        self.store[l].append(record)

    def get_warnings(self):
        return self.store.get("WARNING", []) + []

    def get_errors(self):
        return self.store.get("CRITICAL", []) + self.store.get("ERROR", [])


def setup_logging(config_filename, logging_folder, args):
    level = logging.DEBUG if args.verbose else logging.INFO
    logging_filename = f"{logging_folder}/{config_filename}.log"

    message_store = MessageStore()
    message_store.setLevel(logging.WARNING)
    NOTICE_LEVELV_NUM = 25
    logging.addLevelName(NOTICE_LEVELV_NUM, "NOTICE")

    def notice(self, message, *args, **kws):
        if self.isEnabledFor(NOTICE_LEVELV_NUM):
            self._log(NOTICE_LEVELV_NUM, message, args, **kws)

    logging.Logger.notice = notice
    fmt_verbose = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    fmt_info = "%(message)s"
    fmt = fmt_verbose if args.verbose else fmt_info

    logger = get_logger()

    handlers = [message_store]
    if not args.check:
        handlers.append(logging.FileHandler(logging_filename, mode="w"))
        handlers[-1].setLevel(logging.DEBUG)
        handlers[-1].setFormatter(logging.Formatter(fmt_verbose))
    # logging.basicConfig(level=level, format=fmt, handlers=handlers)

    for h in handlers:
        logger.addHandler(h)

    coloredlogs.install(
        level=level,
        fmt=fmt,
        reconfigure=True,
        level_styles=coloredlogs.parse_encoded_styles(
            "debug=8;notice=green;warning=yellow;error=red,bold;critical=red,inverse"
        ),
    )
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    logger.info(f"Logging streaming out, also saving to {logging_filename}")

    return message_store, logging_filename


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        raw = f.read()
    logging.info("Preprocessing yaml")
    yaml_str = preprocess(raw)
    info = f"# Original input file: {yaml_path}\n# Submitted on login node: {socket.gethostname()}\n"
    yaml_str = info + yaml_str
    config = yaml.safe_load(yaml_str)
    return yaml_str, config


def preprocess(raw):
    lines = raw.split("\n")
    # Get all lines which start with #
    comment_lines = [line[1:] for line in lines if (len(line) > 0) and (line[0] == "#")]
    # Now get all lines which start and end with %
    preprocess_lines = [
        line
        for line in comment_lines
        if (len(line.split()) > 0)
        and (line.split()[0][0] == line.split()[-1][-1] == "%")
    ]
    if len(preprocess_lines) == 0:
        logging.info("No preprocessing found")
        return raw
    logging.debug(f"Found preprocessing lines:\n{preprocess_lines}")
    for preprocess in preprocess_lines:
        # Remove % from preprocess
        preprocess = preprocess.replace("%", "")
        # Run preprocess step
        name, *value = preprocess.split()
        if name == "include:":
            logging.info(f"Including {value[0]}")
            lines = preprocess_include(value, lines)
        else:
            logging.warning(f"Unknown preprocessing step: {name}, skipping")
    yaml_str = "\n".join(lines)
    return yaml_str


def preprocess_include(value, lines):
    include_path = os.path.abspath(os.path.expandvars(value[0]))
    assert os.path.exists(
        include_path
    ), f"Attempting to include {include_path}, but file cannot be found."
    with open(include_path, "r") as f:
        include_yaml = f.read()
    include_yaml = include_yaml.split("\n")
    index = [i for i, l in enumerate(lines) if value[0] in l][0]
    info = [f"# Anchors included from {include_path}"]
    return lines[:index] + info + include_yaml + lines[index + 1 :]


def run(args):
    if args is None:
        return None

    init()

    # Load YAML config file
    yaml_path = os.path.abspath(os.path.expandvars(args.yaml))
    assert os.path.exists(yaml_path), f"File {yaml_path} cannot be found."
    config_raw, config = load_yaml(yaml_path)
    # with open(yaml_path, "r") as f:
    #    config = yaml.safe_load(f)

    overwrites = config.get("GLOBAL")
    if config.get("GLOBALS") is not None:
        logging.warning(
            "Your config file has a GLOBALS section in it. If you're trying to overwrite cfg.yml, rename this to GLOBAL"
        )

    cfg = None
    if overwrites:
        cfg = overwrites.get("CFG_PATH")
    if cfg is None:
        cfg = args.config

    global_config = get_config(initial_path=cfg, overwrites=overwrites)

    config_filename = os.path.basename(args.yaml).split(".")[0].upper()
    output_dir = get_output_dir()
    logging_folder = os.path.abspath(os.path.join(output_dir, config_filename))

    if not args.check:
        mkdirs(logging_folder)
    if os.path.exists(logging_folder):
        chown_dir(logging_folder, walk=args.permission)

    if args.permission:
        return

    message_store, logging_filename = setup_logging(
        config_filename, logging_folder, args
    )

    for i, d in enumerate(global_config["DATA_DIRS"]):
        logging.debug(f"Data directory {i + 1} set as {d}")
        assert (
            d is not None
        ), "Data directory is none, which means it failed to resolve. Check the error message above for why."

    logging.info(
        f"Running on: {os.environ.get('HOSTNAME', '$HOSTNAME not set')} login node."
    )

    manager = Manager(config_filename, yaml_path, config_raw, config, message_store)

    # Gracefully hand Ctrl-c
    def handler(signum, frame):
        logging.error("Ctrl-c was pressed.")
        logging.warning("All remaining tasks will be killed and their hash reset")
        manager.kill_remaining_tasks()
        exit(1)

    signal.signal(signal.SIGINT, handler)

    if args.start is not None:
        args.refresh = True
    manager.set_start(args.start)
    manager.set_finish(args.finish)
    manager.set_force_refresh(args.refresh)
    manager.set_force_ignore_stage(args.ignore)
    num_errs = manager.execute(args.check, args.compress, args.uncompress)
    manager.num_errs = num_errs or 0
    chown_file(logging_filename)
    return manager


def get_syntax(task):
    taskname = [
        "DATAPREP",
        "SIM",
        "LCFIT",
        "CLASSIFY",
        "AGG",
        "MERGE",
        "BIASCOR",
        "CREATE_COV",
        "COSMOFIT",
        "ANALYSE",
    ]
    if task == "options":
        print(f"Possible tasks are: ({[(i, task) for i, task in enumerate(taskname)]})")
        return None

    try:
        task = taskname[int(task)]
    except:
        pass

    assert task in taskname, f"Unknown task {task}"

    base = os.path.dirname(os.path.realpath(__file__))
    with open(f"{base}/docs/src/tasks/{task.lower()}.md", "r") as f:
        print(f.read())

def get_args(test=False):
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "yaml",
        help="the name of the yml config file to run. For example: configs/default.yml",
        type=str,
        nargs="*",
    )
    parser.add_argument(
        "--config",
        help="Location of global config (i.e. cfg.yml)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    parser.add_argument(
        "-s",
        "--start",
        help="Stage to start and force refresh. Accepts either the stage number or name (i.e. 1 or SIM)",
        default=None,
    )
    parser.add_argument(
        "-f",
        "--finish",
        help="Stage to finish at (it runs this stage too). Accepts either the stage number or name (i.e. 1 or SIM)",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--refresh",
        help="Refresh all tasks, do not use hash",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--check",
        help="Check if config is valid",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-p",
        "--permission",
        help="Fix permissions and groups on all output, don't rerun",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-i",
        "--ignore",
        help="Dont rerun tasks with this stage or less. Accepts either the stage number of name (i.e. 1 or SIM)",
        default=None,
    )
    parser.add_argument(
        "-S",
        "--syntax",
        help="Get the syntax of the given stage. Accepts either the stage number or name (i.e. 1 or SIM). If run without argument, will tell you all stage numbers / names.",
        default=None,
        const="options",
        type=str,
        nargs="?",
    )
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument(
        "-C",
        "--compress",
        help="Compress pippin output during job. Combine with -c / --check in order to compress completed pippin job.",
        action="store_true",
        default=False,
    )
    command_group.add_argument(
        "-U",
        "--uncompress",
        help="Do not compress pippin output during job. Combine with -c / --check in order to uncompress completed pippin job.  Mutually exclusive with -C / --compress",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if args.syntax is not None:
        s = args.syntax
        get_syntax(s)
        return None
    elif not test:
        if len(args.yaml) == 0:
            parser.error("You must specify a yaml file!")
        else:
            args.yaml = args.yaml[0]

    return args


if __name__ == "__main__":
    args = get_args()
    if args is not None:
        manager = run(args)
        sys.stdout.flush()
        if manager.num_errs > 0:
            raise (ValueError(f"{manager.num_errs} Errors found"))
