from abc import ABC, abstractmethod
from pippin.config import get_logger, get_hash
import os
import datetime


class Task(ABC):
    FINISHED_SUCCESS = -1
    FINISHED_FAILURE = -9
    logger = get_logger()

    def __init__(self, name, output_dir, dependencies=None):
        self.name = name
        self.output_dir = output_dir
        self.num_jobs = 1
        if dependencies is None:
            dependencies = []
        self.dependencies = dependencies
        self.hash = None
        self.output = {"name": name, "output_dir": output_dir}
        self.hash_file = os.path.join(self.output_dir, "hash.txt")
        self.done_file = os.path.join(self.output_dir, "done.txt")
        self.start_time = None
        self.end_time = None
        self.wall_time = None
        self.stage = None
        self.fresh_run = True
        self.num_empty = 0
        self.num_empty_threshold = 10

    def check_for_job(self, squeue, match):
        if squeue is None:
            return self.num_jobs

        num_jobs = len([i for i in squeue if match in i])
        if num_jobs == 0:
            self.num_empty += 1
            if self.num_empty >= self.num_empty_threshold:
                self.logger.error(f"No more waiting, there are no slurm jobs active that match {match}! Debug output dir {self.output_dir}")
                return Task.FINISHED_FAILURE
            elif self.num_empty > 1:
                self.logger.warning(f"Task {str(self)} has no match for {match} in squeue, warning {self.num_empty}/{self.num_empty_threshold}")
            return 0
        return num_jobs

    def should_be_done(self):
        self.fresh_run = False

    def set_stage(self, stage):
        self.stage = stage

    def get_old_hash(self, quiet=False):
        if os.path.exists(self.hash_file):
            with open(self.hash_file, "r") as f:
                old_hash = f.read().strip()
                if not quiet:
                    self.logger.debug(f"Previous result found, hash is {old_hash}")
                return old_hash
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
        hashes = sorted([dep.get_old_hash(quiet=True) for dep in self.dependencies])
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
        return self._run(force_refresh)

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
        raise ValueError(message)

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
            if result == Task.FINISHED_FAILURE and os.path.exists(self.hash_file):
                os.remove(self.hash_file)
        elif not self.fresh_run:
            self.logger.error("Hash check had passed, so the task should be done, but it said it wasn't!")
            self.logger.error(f"This means it probably crashed, have a look in {self.output_dir}")
            self.logger.error(f"Removing hash from {self.hash_file}")
            if os.path.exists(self.hash_file):
                os.remove(self.hash_file)
            return Task.FINISHED_FAILURE
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
