from abc import ABC, abstractmethod
from pippin.config import get_logger, mkdirs, get_hash
import os


class Task(ABC):
    FINISHED_GOOD = -1
    FINISHED_CRASH = -9

    def __init__(self, name, output_dir):
        self.logger = get_logger()
        self.name = name
        self.output_dir = output_dir
        self.num_jobs = 1
        self.dependencies = []
        self.hash = None
        self.output = {}
        self.hash_file = f"{self.output_dir}/hash.txt"
        mkdirs(self.output_dir)

    def get_old_hash(self):
        if os.path.exists(self.hash_file):
            with open(self.hash_file, "r") as f:
                old_hash = f.read().strip()
                self.logger.debug(f"Previous result found, hash is {old_hash}")
                return old_hash
        else:
            self.logger.debug("No hash found")
        return None

    def get_hash_from_files(self, output_files):
        string_to_hash = ""
        for file in output_files:
            with open(file, "r") as f:
                string_to_hash += f.read()
        new_hash = self.get_hash_from_string(string_to_hash)
        return new_hash

    def get_hash_from_string(self, string_to_hash):
        for dep in self.dependencies:
            string_to_hash += dep.get_old_hash()
        new_hash = get_hash(string_to_hash)
        self.logger.debug(f"Current hash set to {new_hash} from string and dependencies")
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

    @abstractmethod
    def run(self):
        """ Returns the hash of the step if successful, False if not. """
        pass

    @abstractmethod
    def check_completion(self):
        """ Checks if the job is complete or has failed. 
        
        If it is complete it should also load in the any useful results that 
        other tasks may need. 
        
        Such as the location of a trained model or output files.
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__}: {self.name} task, {self.num_jobs} jobs, deps are {[d.name for d in self.dependencies]}"