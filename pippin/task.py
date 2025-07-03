import shutil
from abc import ABC, abstractmethod
from pippin.config import (
    get_logger,
    get_hash,
    ensure_list,
    get_data_loc,
    read_yaml,
    compress_dir,
    uncompress_dir,
)
import tarfile
import os
import datetime
import numpy as np
import yaml
import sys

sys.setrecursionlimit(10000)


class Task(ABC):
    FINISHED_SUCCESS = -1
    FINISHED_FAILURE = -9
    logger = get_logger()

    def __init__(
        self, name, output_dir, dependencies=None, config=None, done_file="done.txt"
    ):
        self.name = name
        self.output_dir = output_dir
        self.num_jobs = 1
        if dependencies is None:
            dependencies = []
        self.dependencies = dependencies
        self.dependents = []
        for task in dependencies:
            task.add_dependent(self)

        if config is None:
            config = {}
        self.config = config.copy()
        self.output = {}

        # Determine if this is an external (already done) job or not
        external_dirs = ensure_list(self.config.get("EXTERNAL_DIRS", []))
        self.logger.debug(f"EXTERNAL_DIRS: {external_dirs}")
        external_names = [os.path.basename(d) for d in external_dirs]
        external_map = self.config.get("EXTERNAL_MAP", {})
        output_name = os.path.basename(output_dir)
        name_match = external_map.get(output_name)
        if external_dirs:
            # This will only trigger if EXTERNAL_MAP is defined and output_name is in external_map
            if name_match is not None:
                matching_dirs = [d for d in external_dirs if name_match in d]
                if len(matching_dirs) == 0:
                    self.logger.error(
                        f"Task {output_name} has external mapping {name_match} but there were no matching EXTERNAL_DIRS"
                    )
                else:
                    if len(matching_dirs) > 1:
                        self.logger.warning(
                            f"Task {output_name} has external mapping {name_match} which matched with multiple EXTERNAL_DIRS: {matching_dirs}. Defaulting to {matching_dirs[0]}"
                        )

                    self.logger.info(f"Found external match for {output_name}")
                    self.config["EXTERNAL"] = matching_dirs[0]
            # If you haven't specified an EXTERNAL_MAP for this output_name, check for exact match
            elif output_name in external_names:
                self.config["EXTERNAL"] = external_dirs[
                    external_names.index(output_name)
                ]
            else:
                self.logger.info(f"No external match found for {output_name}")

        self.external = self.config.get("EXTERNAL")
        if self.external is not None:
            self.logger.debug(f"External config stated to be {self.external}")
            self.external = get_data_loc(self.external)
            # External directory might be compressed
            if not os.path.exists(self.external):
                self.logger.warning(
                    f"External config {self.external} does not exist, checking if it's compressed"
                )
                compressed_dir = self.external + ".tar.gz"
                if not os.path.exists(compressed_dir):
                    self.logger.error(
                        f"{self.external} and {compressed_dir} do not exist"
                    )
                else:
                    self.external = compressed_dir
                    self.logger.debug(
                        f"External config file path resolved to {self.external}"
                    )
                    with tarfile.open(self.external, "r:gz") as tar:
                        for member in tar:
                            if member.isfile():
                                filename = os.path.basename(member.name)
                                if filename != "config.yml":
                                    continue
                                with tar.extractfile(member) as f:
                                    external_config = yaml.load(f, Loader=yaml.Loader)
                                    conf = external_config.get("CONFIG", {})
                                    conf.update(self.config)
                                    self.config = conf
                                    self.output = external_config.get("OUTPUT", {})
                                    self.logger.debug(
                                        "Loaded external config successfully"
                                    )
            else:
                if os.path.isdir(self.external):
                    self.external = os.path.join(self.external, "config.yml")
                self.logger.debug(
                    f"External config file path resolved to {self.external}"
                )
                with open(self.external, "r") as f:
                    external_config = yaml.load(f, Loader=yaml.Loader)
                    conf = external_config.get("CONFIG", {})
                    conf.update(self.config)
                    self.config = conf
                    self.output = external_config.get("OUTPUT", {})
                    self.logger.debug("Loaded external config successfully")

        self.hash = None
        self.hash_file = os.path.join(self.output_dir, "hash.txt")
        self.done_file = os.path.join(self.output_dir, done_file)

        # Info about the job run
        self.start_time = None
        self.end_time = None
        self.wall_time = None
        self.stage = None
        self.fresh_run = True
        self.num_empty = 0
        self.num_empty_threshold = 10
        self.display_threshold = 0
        self.gpu = False

        self.force_refresh = False
        self.force_ignore = False

        self.output.update(
            {
                "name": name,
                "output_dir": output_dir,
                "hash_file": self.hash_file,
                "done_file": self.done_file,
            }
        )
        self.config_file = os.path.join(output_dir, "config.yml")

    def add_dependent(self, task):
        self.dependents.append(task)

    def set_force_refresh(self, force_refresh):
        self.force_refresh = force_refresh

    def set_force_ignore(self, force_ignore):
        self.force_ignore = force_ignore

    def set_setup(self, setup):
        self.task_setup = setup

    def set_sbatch_cpu_header(self, header):
        self.logger.debug("Set cpu header")
        self.sbatch_cpu_header = header

    def set_sbatch_gpu_header(self, header):
        self.logger.debug("Set gpu header")
        self.sbatch_gpu_header = header

    def update_setup(self, setup_dict, task_setup):
        return task_setup.format(**setup_dict)

    def update_header(self, header_dict):
        for key, value in header_dict.items():
            if key in self.sbatch_header:
                self.sbatch_header = self.sbatch_header.replace(key, str(value))
        append_list = header_dict.get("APPEND")
        if append_list is not None:
            lines = self.sbatch_header.split("\n")
            lines += append_list
            self.sbatch_header = "\n".join(lines)
        self.logger.debug("Updated header")

    def clean_header(self, header):
        lines = header.split("\n")
        mask = lambda x: (len(x) > 0) and (x[0] == "#") and ("Sxxx" not in x)
        lines = filter(mask, lines)
        header = "\n".join(lines)
        return header

    def compress(self):
        if os.path.exists(self.output_dir):
            output_file = self.output_dir + ".tar.gz"
            compress_dir(output_file, self.output_dir)
        for t in self.dependencies:
            if os.path.exists(t.output_dir):
                output_file = t.output_dir + ".tar.gz"
                compress_dir(output_file, t.output_dir)

    def uncompress(self):
        source_file = self.output_dir + ".tar.gz"
        if os.path.exists(source_file):
            uncompress_dir(os.path.dirname(self.output_dir), source_file)
        for t in self.dependencies:
            source_file = t.output_dir + ".tar.gz"
            if os.path.exists(source_file):
                uncompress_dir(os.path.dirname(t.output_dir), source_file)

    def _check_regenerate(self, new_hash):
        hash_are_different = new_hash != self.get_old_hash()

        if self.force_ignore:
            if hash_are_different:
                self.logger.warning(
                    f"Warning, hashes are different for {self}, but force_ignore is True so regenerate=False"
                )
            else:
                self.logger.debug(
                    "Hashes agree and force_ignore is set, returning regenerate=False"
                )
            return False
        elif self.force_refresh:
            self.logger.debug("Force refresh is set, returning regenerate=True")
            return True
        else:
            if hash_are_different:
                self.logger.debug("Hashes are different, regenerating")
                return True
            else:
                self.logger.debug("Hashes are the same, not regenerating")
                return False

    def write_config(self):
        content = {"CONFIG": self.config, "OUTPUT": self.output}
        with open(self.config_file, "w") as f:
            yaml.dump(content, f, sort_keys=False)

    def load_config(self):
        with open(self.config_file, "r") as f:
            content = yaml.safe_load(f)
            return content

    def clear_config(self):
        if os.path.exists(self.config_file):
            os.remove(self.config_file)

    def clear_hash(self):
        if os.path.exists(self.hash_file):
            ext = os.path.splitext(self.hash_file)
            old_hash_path = os.path.join(ext[0] + "_old" + ext[-1])
            self.logger.debug(f"Copied {self.hash_file} to {old_hash_path}")
            shutil.copyfile(self.hash_file, old_hash_path)
            self.logger.debug(f"Removing {self.hash_file}")
            os.remove(self.hash_file)
        self.clear_config()

    def check_for_job(self, squeue, match):
        if squeue is None:
            return self.num_jobs

        num_jobs = len([i for i in squeue if match in i])
        if num_jobs == 0:
            self.num_empty += 1
            if self.num_empty >= self.num_empty_threshold:
                self.logger.error(
                    f"No more waiting, there are no slurm jobs active that match {match}! Debug output dir {self.output_dir}"
                )
                return Task.FINISHED_FAILURE
            elif self.num_empty > 1 and self.num_empty > self.display_threshold:
                self.logger.warning(
                    f"Task {str(self)} has no match for {match} in squeue, warning {self.num_empty}/{self.num_empty_threshold}"
                )
            return 0
        return num_jobs

    def should_be_done(self):
        self.fresh_run = False

    def set_stage(self, stage):
        self.stage = stage

    def get_old_hash(self, quiet=False, required=False):
        if os.path.exists(self.hash_file):
            with open(self.hash_file, "r") as f:
                old_hash = f.read().strip()
                if not quiet:
                    self.logger.debug(f"Previous result found, hash is {old_hash}")
                return old_hash
        else:
            if required:
                self.logger.error(f"No hash found for {self} in {self.hash_file}")
            else:
                self.logger.debug(f"No hash found for {self}")
        return "_NONE_"

    def get_hash_from_files(self, output_files):
        string_to_hash = ""
        for file in output_files:
            with open(file, "r") as f:
                string_to_hash += f.read()
        new_hash = self.get_hash_from_string(string_to_hash)
        return new_hash

    def get_hash_from_string(self, string_to_hash):
        hashes = sorted(
            [dep.get_old_hash(quiet=True, required=True) for dep in self.dependencies]
        )
        string_to_hash += " ".join(hashes)
        new_hash = get_hash(string_to_hash)
        self.logger.debug(f"Current hash set to {new_hash}")
        return new_hash

    def save_new_hash(self, new_hash):
        with open(self.hash_file, "w") as f:
            f.write(str(new_hash))
            self.logger.debug(f"New hash {new_hash}")
            self.logger.debug(f"New hash saved to {self.hash_file}")

    def set_num_jobs(self, num_jobs):
        self.num_jobs = num_jobs

    def add_dependency(self, task):
        self.dependencies.append(task)

    def run(self):
        self.uncompress()
        if self.external is not None:
            self.logger.debug(f"Name: {self.name} External: {self.external}")
            if os.path.exists(self.output_dir) and not self.force_refresh:
                self.logger.info(
                    f"Not copying external site, output_dir already exists at {self.output_dir}"
                )
            else:
                if os.path.exists(self.output_dir):
                    self.logger.debug(f"Removing old directory {self.output_dir}")
                    shutil.rmtree(self.output_dir, ignore_errors=True)
                if ".tar.gz" in self.external:
                    tardir = os.path.basename(self.external).replace(".tar.gz", "")
                    self.logger.info(
                        f"Copying files from {self.external} to {self.output_dir}"
                    )

                    shutil.copyfile(self.external, self.output_dir + ".tar.gz")
                    self.uncompress()
                    shutil.move(
                        os.path.join(os.path.dirname(self.output_dir), tardir),
                        self.output_dir,
                    )
                else:
                    self.logger.info(
                        f"Copying from {os.path.dirname(self.external)} to {self.output_dir}"
                    )
                    shutil.copytree(
                        os.path.dirname(self.external), self.output_dir, symlinks=True
                    )
            return True

        return self._run()

    def scan_file_for_error(self, path, *error_match, max_lines=10):
        assert len(error_match) >= 1, (
            "You need to specify what string to search for. I have nothing."
        )
        found = False
        if not os.path.exists(path):
            self.logger.warning(f"Note, expected log path {path} does not exist")
            return False

        with open(path) as f:
            for i, line in enumerate(f.read().splitlines()):
                error_found = np.any([e in line for e in error_match])
                if error_found:
                    index = i
                    found = True
                    self.logger.error(f"Found error in file {path}, excerpt below")
                if found and i - index <= max_lines:
                    self.logger.error(f"Excerpt:    {line}")
        return found

    def scan_files_for_error(
        self, paths, *error_match, max_lines=10, max_erroring_files=3
    ):
        num_errors = 0
        self.logger.debug(f"Found {len(paths)} to scan")
        for path in paths:
            if "FAIL_SUMMARY.LOG" in path.upper():
                self.logger.debug(f"Found {path}, loading in YAML contents")
                fail_summary = read_yaml(path)
                for key, dicts in fail_summary.items():
                    if key.startswith("FAILURE-0"):
                        self.logger.error(
                            f"{key}: {' '.join(dicts.get('ABORT_MESSAGES', 'Unknown message'))}"
                        )
                        self.logger.error(
                            f"{key}: Detailed in {dicts.get('JOB_LOG_FILE', 'Unknown path')}"
                        )
                        num_errors += 1
                    if num_errors > max_erroring_files:
                        break
            else:
                self.logger.debug(f"Scanning {path} for error")
                if self.scan_file_for_error(path, *error_match, max_lines=max_lines):
                    num_errors += 1
            if num_errors >= max_erroring_files:
                break
        return num_errors > 0

    @staticmethod
    def match_tasks(mask, deps, match_none=True, allowed_failure=False):
        if mask is None:
            if match_none:
                mask = ""
            else:
                return []
        if isinstance(mask, str):
            if mask == "*":
                mask = ""
        mask = ensure_list(mask)

        matching_deps = [d for d in deps if any(x in d.name for x in mask)]

        for m in mask:
            specific_match = [d for d in matching_deps if m in d.name]
            if len(specific_match) == 0 and not allowed_failure:
                Task.fail_config(
                    f"Mask '{m}' does not match any deps. Probably a typo. Available options are {deps}"
                )

        return matching_deps

    @staticmethod
    def match_tasks_of_type(mask, deps, *cls, match_none=True, allowed_failure=False):
        return Task.match_tasks(
            mask,
            Task.get_task_of_type(deps, *cls),
            match_none=match_none,
            allowed_failure=allowed_failure,
        )

    @abstractmethod
    def _run(self):
        """Execute the primary function of the task

        :param force_refresh: to force refresh and rerun - do not pass hash checks
        :return: true or false if the job launched successfully
        """
        pass

    @staticmethod
    def get_task_of_type(tasks, *cls):
        return [t for t in tasks if isinstance(t, tuple(cls))]

    @staticmethod
    def fail_config(message):
        Task.logger.error(message)
        raise ValueError("Task failed config")

    @staticmethod
    @abstractmethod
    def get_tasks(
        config, prior_tasks, base_output_dir, stage_number, prefix, global_config
    ):
        raise NotImplementedError()

    def get_wall_time_str(self):
        if self.end_time is not None and self.start_time is not None:
            return str(datetime.timedelta(seconds=self.wall_time))
        return None

    def check_completion(self, squeue):
        """Checks if the job has completed.

        Invokes  `_check_completion` and determines wall time.

        :return: Task.FINISHED_SUCCESS, Task.FNISHED_FAILURE or the number of jobs still running
        """
        result = self._check_completion(squeue)
        if result in [Task.FINISHED_SUCCESS, Task.FINISHED_FAILURE]:
            if os.path.exists(self.done_file):
                self.end_time = os.path.getmtime(self.done_file)
                if self.start_time is None and os.path.exists(self.hash_file):
                    self.start_time = os.path.getmtime(self.hash_file)
                if self.end_time is not None and self.start_time is not None:
                    self.wall_time = int(
                        self.end_time - self.start_time + 0.5
                    )  # round up
                    self.logger.info(
                        f"Task finished with wall time {self.get_wall_time_str()}"
                    )
            if result == Task.FINISHED_FAILURE:
                self.clear_hash()
        elif not self.fresh_run:
            self.logger.error(
                "Hash check had passed, so the task should be done, but it said it wasn't!"
            )
            self.logger.error(
                f"This means it probably crashed, have a look in {self.output_dir}"
            )
            self.logger.error(f"Removing hash from {self.hash_file}")
            self.clear_hash()
            # TODO try rerunning task
            return Task.FINISHED_FAILURE
        if (
            self.external is None
            and result == Task.FINISHED_SUCCESS
            and not os.path.exists(self.config_file)
        ):
            self.write_config()
        return result

    @abstractmethod
    def _check_completion(self, squeue):
        """Checks if the job is complete or has failed.

        If it is complete it should also load in the any useful results that
        other tasks may need in `self.output` dictionary

        Such as the location of a trained model or output files.
        :param squeue:
        """
        pass

    def __str__(self):
        wall_time = self.get_wall_time_str()
        if wall_time is not None:
            extra = f"wall time {wall_time}, "
        else:
            extra = ""
        if len(self.dependencies) > 5:
            deps = f"{[d.name for d in self.dependencies[:5]]} + {len(self.dependencies) - 5} more deps"
        else:
            deps = f"{[d.name for d in self.dependencies]}"

        if self.external is None:
            return f"{self.__class__.__name__} {self.name} task ({extra}{self.num_jobs} jobs, deps {deps})"
        else:
            return f"{self.__class__.__name__} {self.name} task (EXTERNAL JOB, deps {deps})"

    def __repr__(self):
        return self.__str__()

    def get_dep(self, *clss, fail=False):
        for d in self.dependencies:
            for cls in clss:
                if isinstance(d, cls):
                    return d
        if fail:
            raise ValueError(f"No deps have class of type {clss}")
        return None

    def get_deps(self, *clss):
        return [d for d in self.dependencies if isinstance(d, tuple(clss))]
