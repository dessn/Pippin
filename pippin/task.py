from abc import ABC, abstractmethod
from pippin.config import get_logger, mkdirs


class Task(ABC):
    def __init__(self, name, output_dir):
        self.logger = get_logger()
        self.name = name
        self.output_dir = output_dir
        self.num_jobs = 1
        self.dependencies = []
        self.hash = None
        mkdirs(self.output_dir)

    def set_num_jobs(self, num_jobs):
        self.num_jobs = num_jobs

    def add_dependency(self, task):
        self.dependencies.append(task)

    def get_hash(self):
        if self.hash is None:
            self.hash = self._get_hash()
        return self.hash

    #@abstractmethod
    def _get_hash(self):
        pass

    @abstractmethod
    def run(self):
        """ Returns the hash of the step if successful, False if not. """
        pass

    @abstractmethod
    def check_completion(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}: {self.name} task, {self.num_jobs} jobs, deps are {[d.name for d in self.dependencies]}"