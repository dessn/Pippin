import configparser
import inspect
import logging
import hashlib
import os
import shutil
import stat


def singleton(fn):
    instance = None

    def get(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = fn(*args, **kwargs)
        return instance

    return get


@singleton
def get_config(initial_path=None):
    if initial_path is None:
        filename = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/../cfg.ini")
    else:
        filename = initial_path
    config = configparser.ConfigParser()
    config.read(filename)
    return config


def get_output_dir():
    output_dir = get_config()["OUTPUT"]["output_dir"]
    if "$" in output_dir:
        output_dir = os.path.expandvars(output_dir)
        if "$" in output_dir:
            raise ValueError(f"Could not resolve variable in path: {output_dir}")
    if not output_dir.startswith("/"):
        output_dir = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/../" + output_dir)
    return output_dir


def get_data_loc(data_dir, path):
    if "$" in path:
        path = os.path.expandvars(path)
    if path.startswith("/"):
        return path
    else:
        return os.path.join(data_dir, path)


def get_output_loc(path):
    if "$" in path:
        path = os.path.expandvars(path)
    if path.startswith("/"):
        return path
    else:
        return os.path.join(get_output_dir(), path)


def get_hash(input_string):
    return hashlib.sha256(input_string.encode("utf-8")).hexdigest()


@singleton
def get_logger():
    return logging.getLogger("pippin")


def mkdirs(path):
    # Do this layer by layer so we can chown correctly
    if not os.path.exists(path):
        parent = os.path.dirname(path)
        mkdirs(parent)
        os.makedirs(path, exist_ok=True, mode=0o7701)
        chown_dir(path)


def copytree(src, dst, symlinks=False, ignore=None):
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
                os.symlink(os.readlink(s), d)
                try:
                    st = os.lstat(s)
                    mode = stat.S_IMODE(st.st_mode)
                    os.lchmod(d, mode)
                except:
                    pass  # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def chown_file(path):
    import grp

    global_config = get_config()
    logger = get_logger()
    groupinfo = grp.getgrnam(global_config["SNANA"]["group"])
    group_id = groupinfo.gr_gid
    try:
        os.chown(path, -1, group=group_id, follow_symlinks=False)
    except Exception:
        logger.debug(f"Chown error: {path}")


def chown_dir(directory):
    import grp

    global_config = get_config()
    logger = get_logger()
    groupinfo = grp.getgrnam(global_config["SNANA"]["group"])
    group_id = groupinfo.gr_gid
    try:
        shutil.chown(directory, group=global_config["SNANA"]["group"])
        os.chmod(directory, 0o770)
    except Exception as e:
        logger.exception(f"Chown error: {directory}")
        return
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            if not os.path.islink(os.path.join(root, d)):
                try:
                    os.chown(os.path.join(root, d), -1, group_id, follow_symlinks=False)
                    os.chmod(os.path.join(root, d), 0o770)
                except Exception as e:
                    logger.warning(f"Chown error: {os.path.join(root, d)}")
        for f in files:
            if not os.path.islink(os.path.join(root, f)):
                try:
                    os.chown(os.path.join(root, f), -1, group_id, follow_symlinks=False)
                    os.chmod(os.path.join(root, f), 0o660)
                except Exception as e:
                    logger.warning(f"Chown error: {os.path.join(root, f)}")


def ensure_list(a):
    if isinstance(a, list):
        return a
    return [a]


if __name__ == "__main__":
    c = get_config()
    print(c.sections())
    print(c.get("SNANA", "sim_dir"))
    print(c["OUTPUT"].getint("ping_frequency"))
