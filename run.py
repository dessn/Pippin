import argparse
import os
import yaml
import logging
import coloredlogs
from pippin.config import mkdirs, get_logger, get_output_dir, chown_file, get_config, chown_dir
from pippin.manager import Manager
from colorama import init

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
    fmt = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s" if args.verbose else "%(message)s"
    handlers = [logging.StreamHandler(), message_store]
    handlers[0].setLevel(level)
    if not args.check:
        handlers.append(logging.FileHandler(logging_filename, mode="w"))
        handlers[-1].setLevel(logging.DEBUG)
    logging.basicConfig(level=level, format=fmt, handlers=handlers)

    coloredlogs.install(
        level=level,
        fmt=fmt,
        reconfigure=True,
        level_styles=coloredlogs.parse_encoded_styles("debug=8;notice=green;warning=yellow;error=red,bold;critical=red,inverse"),
    )
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    logger = get_logger()
    logger.info(f"Logging streaming out, also saving to {logging_filename}")

    return message_store, logging_filename


def run(args):

    if args is None:
        return None

    init()

    # Load YAML config file
    yaml_path = os.path.abspath(os.path.expandvars(args.yaml))
    assert os.path.exists(yaml_path), f"File {yaml_path} cannot be found."
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    overwrites = config.get("GLOBAL")
    if config.get("GLOBALS") is not None:
        logging.warning("Your config file has a GLOBALS section in it. If you're trying to overwrite cfg.yml, rename this to GLOBAL")

    cfg = None
    if config.get("GLOBAL"):
        cfg = config.get("GLOBAL").get("CFG_PATH")
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

    message_store, logging_filename = setup_logging(config_filename, logging_folder, args)

    for i, d in enumerate(global_config["DATA_DIRS"]):
        logging.debug(f"Data directory {i + 1} set as {d}")
        assert d is not None, "Data directory is none, which means it failed to resolve. Check the error message above for why."

    logging.info(f"Running on: {os.environ.get('HOSTNAME', '$HOSTNAME not set')} login node.")

    manager = Manager(config_filename, yaml_path, config, message_store)
    if args.start is not None:
        args.refresh = True
    manager.set_start(args.start)
    manager.set_finish(args.finish)
    manager.set_force_refresh(args.refresh)
    manager.set_force_ignore_stage(args.ignore)
    manager.execute(args.check)
    chown_file(logging_filename)
    return manager

def get_syntax():
    syntax = {}
    base = os.path.dirname(os.path.realpath(__file__))
    with open(f"{base}/README.md", 'r') as f:
        readme = f.read()
    lines = readme.split('\n')
    start, end = [idx for (idx, line) in enumerate(lines) if "[//]" in line]
    lines = lines[start:end]
    index = [idx for (idx, line) in enumerate(lines) if "###" == line.split(' ')[0]]
    tasks = []
    for i in range(len(index)):
        idx = index[i]
        if idx != index[-1]:
            tasks.append("\n".join(lines[idx+2:index[i+1]-1]))
        else:
            tasks.append("\n".join(lines[idx+2:-1]))
    taskname = ["DATAPREP", "SIM", "LCFIT", "CLASSIFY", "AGG", "MERGE", "BIASCOR", "CREATE_COV", "COSMOMC", "ANALYSE"]
    for i, name in enumerate(taskname):
        syntax[name] = tasks[i]
    syntax["options"] = f"Possible tasks are: ({[(i, task) for i, task in enumerate(taskname)]})"
    return syntax

def print_syntax(s):
    syntax = get_syntax()
    try:
        keys = list(syntax.keys())
        s = int(s)
        if s < 0 or s > len(keys) - 1:
            raise ValueError(f"Unknown task number {s}")
        key = keys[s]
    except ValueError:
        key = s
    if key not in syntax.keys():
        raise ValueError(f"Unknown task {key}")
    msg = syntax[key]
    print(msg)
    return None

def get_args(test=False):
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", help="the name of the yml config file to run. For example: configs/default.yml", type=str, nargs='*')
    parser.add_argument("--config", help="Location of global config", default=None, type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-s", "--start", help="Stage to start and force refresh", default=None)
    parser.add_argument("-f", "--finish", help="Stage to finish at (it runs this stage too)", default=None)
    parser.add_argument("-r", "--refresh", help="Refresh all tasks, do not use hash", action="store_true")
    parser.add_argument("-c", "--check", help="Check if config is valid", action="store_true", default=False)
    parser.add_argument("-p", "--permission", help="Fix permissions and groups on all output, don't rerun", action="store_true", default=False)
    parser.add_argument("-i", "--ignore", help="Dont rerun tasks with this stage or less", default=None)
    parser.add_argument("-S", "--syntax", help="Get the syntax of the given task.", default=None, const="options", type=str, nargs='?')
    args = parser.parse_args()

    if args.syntax is not None:
        s = args.syntax
        print_syntax(s)
        return None
    elif not test:
        if len(args.yaml) == 0:
            parser.error("You must specify a yaml file!")
        else:
            args.yaml = args.yaml[0]

    return args


if __name__ == "__main__":
    args = get_args()
    run(args)
