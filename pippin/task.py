import logging
import shutil
from abc import ABC, abstractmethod
from pippin.config import get_logger, get_hash, ensure_list, get_data_loc
import os
import datetime
import numpy as np
import yaml


class Task(ABC):
    FINISHED_SUCCESS = -1
    FINISHED_FAILURE = -9
    logger = get_logger()

    def __init__(self, name, output_dir, dependencies=None, config=None, done_file="done.txt"):
        self.name = name
        self.output_dir = output_dir
        self.num_jobs = 1
        if dependencies is None:
            dependencies = []
        self.dependencies = dependencies

        if config is None:
            config = {}
        self.config = config
        self.output = {}

        # Determine if this is an external (already done) job or not
        external_dirs = config.get("EXTERNAL_DIRS", [])
        external_names = [os.path.basename(d) for d in external_dirs]
        output_name = os.path.basename(output_dir)
        print("DEBUG OUTPUT: ", external_dirs, external_names, output_name, external_dirs)
        if external_dirs:
            if output_name in external_names:
                print("DEBUG 2: ", external_names.index(output_name), external_dirs[external_names.index(output_name)])
                config["EXTERNAL"] = external_dirs[external_names.index(output_name)]
        self.external = config.get("EXTERNAL")
        if self.external is not None:
            logging.debug(f"External config stated to be {self.external}")
            self.external = get_data_loc(self.external)
            if os.path.isdir(self.external):
                self.external = os.path.join(self.external, "config.yml")
            logging.debug(f"External config file path resolved to {self.external}")
            with open(self.external, "r") as f:
                external_config = yaml.safe_load(f)
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

        self.output.update({"name": name, "output_dir": output_dir, "hash_file": self.hash_file, "done_file": self.done_file})
        self.config_file = os.path.join(output_dir, "config.yml")

    def write_config(self):
        content = {"CONFIG": self.config, "OUTPUT": self.output}
        with open(self.config_file, "w") as f:
            yaml.dump(content, f)

    def load_config(self):
        with open(self.config_file, "r") as f:
            content = yaml.safe_load(f)
            return content

    def clear_config(self):
        if os.path.exists(self.config_file):
            os.remove(self.config_file)

    def clear_hash(self):
        if os.path.exists(self.hash_file):
            os.remove(self.hash_file)
        self.clear_config()

    def check_for_job(self, squeue, match):
        if squeue is None:
            return self.num_jobs

        num_jobs = len([i for i in squeue if match in i])
        if num_jobs == 0:
            self.num_empty += 1
            if self.num_empty >= self.num_empty_threshold:
                self.logger.error(f"No more waiting, there are no slurm jobs active that match {match}! Debug output dir {self.output_dir}")
                return Task.FINISHED_FAILURE
            elif self.num_empty > 1 and self.num_empty > self.display_threshold:
                self.logger.warning(f"Task {str(self)} has no match for {match} in squeue, warning {self.num_empty}/{self.num_empty_threshold}")
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
                self.logger.error(f"No hash found for {self}")
            else:
                self.logger.debug(f"No hash found for {self}")
        return None

    def get_hash_from_files(self, output_files):
        string_to_hash = ""
        for file in output_files:
            with open(file, "r") as f:
                string_to_hash += f.read()
        new_hash = self.get_hash_from_string(string_to_hash)
        return new_hash

    def get_hash_from_string(self, string_to_hash):
        hashes = sorted([dep.get_old_hash(quiet=True, required=True) for dep in self.dependencies])
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

    def run(self, force_refresh):
        if self.external is not None:
            if os.path.exists(self.output_dir) and not force_refresh:
                self.logger.info(f"Not copying external site, output_dir already exists at {self.output_dir}")
            else:
                self.logger.info(f"Copying from {os.path.dirname(self.external)} to {self.output_dir}")
                shutil.copytree(os.path.dirname(self.external), self.output_dir, symlinks=True)
            return True

        return self._run(force_refresh)

    def scan_file_for_error(self, path, *error_match, max_lines=10):
        assert len(error_match) >= 1, "You need to specify what string to search for. I have nothing."
        found = False
        if not os.path.exists(path):
            self.logger.warning(f"Note, expected log path {path} does not exist")
            return False

        with open(path) as f:
            for i, line in enumerate(f.read().splitlines()):
                error_found = np.any([e in line for e in error_match])
                index = i
                if error_found:
                    found = True
                    self.logger.error(f"Found error in file {path}, excerpt below")
                if found and i - index <= max_lines:
                    self.logger.error(f"Excerpt:    {line}")
        return found

    def scan_files_for_error(self, paths, *error_match, max_lines=10, max_erroring_files=3):
        num_errors = 0
        self.logger.debug(f"Found {len(paths)} to scan")
        for path in paths:
            self.logger.debug(f"Scanning {path} for error")
            if self.scan_file_for_error(path, *error_match, max_lines=max_lines):
                num_errors += 1
            if num_errors >= max_erroring_files:
                break
        return num_errors > 0

    @staticmethod
    def match_tasks(mask, deps, match_none=True):
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
            if len(specific_match) == 0:
                Task.fail_config(f"Mask '{m}' does not match any deps. Probably a typo. Available options are {deps}")

        return matching_deps

    @staticmethod
    def match_tasks_of_type(mask, deps, *cls, match_none=True):
        return Task.match_tasks(mask, Task.get_task_of_type(deps, *cls), match_none=match_none)

    @abstractmethod
    def _run(self, force_refresh):
        """ Execute the primary function of the task

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
        raise ValueError(f"Task failed config")

    @staticmethod
    @abstractmethod
    def get_tasks(config, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        raise NotImplementedError()

    def get_wall_time_str(self):
        if self.end_time is not None and self.start_time is not None:
            return str(datetime.timedelta(seconds=self.wall_time))
        return None

    def check_completion(self, squeue):
        """ Checks if the job has completed.

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
                    self.wall_time = int(self.end_time - self.start_time + 0.5)  # round up
                    self.logger.info(f"Task finished with wall time {self.get_wall_time_str()}")
            if result == Task.FINISHED_FAILURE:
                self.clear_hash()
        elif not self.fresh_run:
            self.logger.error("Hash check had passed, so the task should be done, but it said it wasn't!")
            self.logger.error(f"This means it probably crashed, have a look in {self.output_dir}")
            self.logger.error(f"Removing hash from {self.hash_file}")
            self.clear_hash()
            return Task.FINISHED_FAILURE
        if self.external is None and result == Task.FINISHED_SUCCESS and not os.path.exists(self.config_file):
            self.write_config()
        return result

    @abstractmethod
    def _check_completion(self, squeue):
        """ Checks if the job is complete or has failed. 
        
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
        return f"{self.__class__.__name__} {self.name} task ({extra}{self.num_jobs} jobs, deps {deps})"

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
