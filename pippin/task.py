from abc import ABC, abstractmethod
from pippin.config import get_logger, get_hash
import os
import datetime


class Task(ABC):
    FINISHED_SUCCESS = -1
    FINISHED_FAILURE = -9

    def __init__(self, name, output_dir, dependencies=None):
        self.logger = get_logger()
        self.name = name
        self.output_dir = output_dir
        self.num_jobs = 1
        if dependencies is None:
            dependencies = []
        self.dependencies = dependencies
        self.hash = None
        self.output = {
            "name": name,
            "output_dir": output_dir
        }
        self.hash_file = os.path.join(self.output_dir, "hash.txt")
        self.done_file = os.path.join(self.output_dir, "done.txt")
        self.start_time = None
        self.end_time = None
        self.wall_time = None

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
        print(string_to_hash)
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
        """ Returns the hash of the step if successful, False if not. """
        pass

    def get_wall_time_str(self):
        if self.end_time is not None and self.start_time is not None:
            return str(datetime.timedelta(seconds=self.wall_time))
        return None

    def check_completion(self):
        result = self._check_completion()
        if result in [Task.FINISHED_SUCCESS, Task.FINISHED_FAILURE]:
            if os.path.exists(self.done_file):
                self.end_time = os.path.getmtime(self.done_file)
                if self.start_time is None and os.path.exists(self.hash_file):
                    self.start_time = os.path.getmtime(self.hash_file)
                if self.end_time is not None and self.start_time is not None:
                    self.wall_time = int(self.end_time - self.start_time + 0.5)  # round up
                    self.logger.info(f"Task finished with wall time {self.get_wall_time_str()}")
        return result

    @abstractmethod
    def _check_completion(self):
        """ Checks if the job is complete or has failed. 
        
        If it is complete it should also load in the any useful results that 
        other tasks may need. 
        
        Such as the location of a trained model or output files.
        """
        pass

    def __str__(self):
        wall_time = self.get_wall_time_str()
        if wall_time is not None:
            extra = f"wall time {wall_time}, "
        else:
            extra = ""
        return f"{self.__class__.__name__} {self.name} task ({extra}{self.num_jobs} jobs, deps {[d.name for d in self.dependencies]})"
