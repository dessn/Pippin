import os
import inspect
import logging
import shutil
import subprocess
import time
import tempfile

from pippin.config import get_hash, chown_dir, copytree, mkdirs
from pippin.task import Task


class Aggregator(Task):
    def __init__(self, name, output_dir):
        super().__init__(name, output_dir)
        self.passed = False

    def check_completion(self):
        return Task.FINISHED_GOOD if self.passed else Task.FINISHED_CRASH

    def run(self):
        pass
        # For each dependency, attempt to find the predictions file
        # Assign column names and aggregate
        # Set output to file name and a column map