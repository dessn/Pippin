from run import run, get_args


def get_manager(**kwargs):
    args = get_args(test=True)
    args.config = "tests/config_files/cfg_dev.yml"
    for key, value in kwargs.items():
        setattr(args, key, value)
    manager = run(args)
    return manager
