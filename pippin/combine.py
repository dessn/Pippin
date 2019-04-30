from pippin.task import Task


class Aggregator(Task):
    def __init__(self, name, output_dir, dependencies, options):
        super().__init__(name, output_dir, dependencies=dependencies)
        self.options = options
        self.passed = False
        self.cmd = ["combine_fitres.exe", "t", "-outprefix", "merged"]

    def _check_completion(self):
        return Task.FINISHED_SUCCESS if self.passed else Task.FINISHED_FAILURE

    def _run(self, force_refresh):
        return True

