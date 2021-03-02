import os
import shutil
import subprocess
import time
from colorama import Fore, Style

from pippin.aggregator import Aggregator
from pippin.analyse import AnalyseChains
from pippin.biascor import BiasCor
from pippin.classifiers.classifier import Classifier
from pippin.config import get_logger, get_config, get_output_dir, mkdirs, chown_dir, chown_file, get_data_loc
from pippin.cosmomc import CosmoMC
from pippin.create_cov import CreateCov
from pippin.dataprep import DataPrep
from pippin.merge import Merger
from pippin.snana_fit import SNANALightCurveFit
from pippin.snana_sim import SNANASimulation
from pippin.task import Task


class Manager:
    task_order = [DataPrep, SNANASimulation, CreateHeatmaps, SNANALightCurveFit, Classifier, Aggregator, Merger, BiasCor, CreateCov, CosmoMC, AnalyseChains]
    stages = ["DATAPREP", "SIM", "CREATE_HEATMAPS", "LCFIT", "CLASSIFY", "AGGREGATE", "MERGE", "BIASCOR", "CREATE_COV", "COSMOMC", "ANALYSE"]

    def __init__(self, filename, config_path, config, message_store):
        self.logger = get_logger()
        self.task_index = {t: i for i, t in enumerate(self.task_order)}
        self.message_store = message_store
        self.filename = filename
        self.filename_path = config_path
        self.run_config = config
        self.global_config = get_config()

        self.prefix = self.global_config["QUEUE"]["prefix"] + "_" + filename
        self.max_jobs = int(self.global_config["QUEUE"]["max_jobs"])
        self.max_jobs_gpu = int(self.global_config["QUEUE"]["max_gpu_jobs"])
        self.max_jobs_in_queue = int(self.global_config["QUEUE"]["max_jobs_in_queue"])
        self.max_jobs_in_queue_gpu = int(self.global_config["QUEUE"]["max_gpu_jobs_in_queue"])

        self.logger.debug(self.global_config.keys())

        self.sbatch_cpu_path = get_data_loc(self.global_config["SBATCH"]["cpu_location"])
        with open(self.sbatch_cpu_path, 'r') as f:
            self.sbatch_cpu_header = f.read()
        self.sbatch_gpu_path = get_data_loc(self.global_config["SBATCH"]["gpu_location"])
        with open(self.sbatch_gpu_path, 'r') as f:
            self.sbatch_gpu_header = f.read()
        self.sbatch_clean = self.global_config["SBATCH"]["gpu_location"]
        if self.sbatch_clean:
            self.sbatch_cpu_header = self.clean_header(self.sbatch_cpu_header)
            self.sbatch_gpu_header = self.clean_header(self.sbatch_gpu_header)
        self.setup_task_location = self.global_config["SETUP"]["location"]
        self.load_task_setup()

        self.output_dir = os.path.join(get_output_dir(), self.filename)
        self.tasks = None
        self.num_jobs_queue = 0
        self.num_jobs_queue_gpu = 0

        self.start = None
        self.finish = None
        self.force_refresh = False
        self.force_ignore_stage = None

    def load_task_setup(self):
        tasks = ['cosmomc', 'snirf', 'analyse', 'supernnova', 'nearest_neighbour', 'create_cov']
        self.task_setup = {}
        for task in tasks:
            with open(get_data_loc(f"{self.setup_task_location}/{task}"), 'r') as f:
                self.task_setup[task] = f.read()

    def get_force_refresh(self, task):
        if self.start is None:
            return self.force_refresh
        index = None
        for i, t in enumerate(self.task_order):
            if isinstance(task, t):
                index = i
        if index is None:
            self.logger.error(f"Task {task} did not match any class in the task order!")
            index = 0
        force = index >= self.start
        self.logger.debug(f"Start set! Task {task} has index {index}, start index set {self.start}, so returning {force}")
        return force

    def get_force_ignore(self, task):
        if self.force_ignore_stage is None:
            return False
        index = None
        for i, t in enumerate(self.task_order):
            if isinstance(task, t):
                index = i
        if index is None:
            self.logger.error(f"Task {task} did not match any class in the task order!")
            assert index is not None
        force_ignore = index <= self.force_ignore_stage
        self.logger.debug(f"Task {task} has index {index}, ignore index is {self.force_ignore_stage}, so returning force_ignore={force_ignore}")
        return force_ignore

    def set_force_refresh(self, force_refresh):
        self.force_refresh = force_refresh

    def set_force_ignore_stage(self, force_ignore_stage):
        self.force_ignore_stage = self.resolve_stage(force_ignore_stage)

    def clean_header(self, header):
        lines = header.split('\n')
        mask = lambda x: (len(x) > 0) and (x[0] == '#') and ('xxxx' not in x)
        lines = filter(mask, lines)
        header = '\n'.join(lines)
        return header

    def set_start(self, stage):
        self.start = self.resolve_stage(stage)

    def set_finish(self, stage):
        self.finish = self.resolve_stage(stage)

    def resolve_stage(self, stage):
        if stage is None:
            return None
        if stage.isdigit():
            num = int(stage)
        else:
            key = stage.upper()
            assert key in Manager.stages, f"Stage {key} is not in recognised keys {Manager.stages}"
            num = Manager.stages.index(key)
        assert 0 <= num < len(Manager.stages), f"Stage {num} is not in recognised values is not valid - from 0 to {len(Manager.stages) - 1}"
        return num

    def get_tasks(self, config):

        total_tasks = []
        try:
            for i, task in enumerate(Manager.task_order):
                if self.finish is None or i <= self.finish:
                    new_tasks = task.get_tasks(config, total_tasks, self.output_dir, i, self.prefix, self.global_config)
                    if new_tasks is not None:
                        total_tasks += new_tasks
        except Exception as e:
            self.logger.exception(e, exc_info=False)
            raise e
        self.logger.info("")
        self.logger.notice("Listing tasks:")
        for task in total_tasks:
            self.logger.notice(f"\t{task}")
        self.logger.info("")
        return total_tasks

    def get_num_running_jobs(self):
        num_jobs = int(subprocess.check_output("squeue -ho %A -u $USER | wc -l", shell=True, stderr=subprocess.STDOUT))
        return num_jobs

    def get_task_to_run(self, tasks_to_run, done_tasks):
        for t in tasks_to_run:
            can_run = True
            for dep in t.dependencies:
                if dep not in done_tasks:
                    can_run = False
            if t.gpu and self.num_jobs_queue_gpu + t.num_jobs >= self.max_jobs_in_queue_gpu:
                self.logger.warning(f"Cant submit {t} because GPU NUM_JOBS {t.num_jobs} would exceed {self.num_jobs_queue_gpu}/{self.max_jobs_in_queue_gpu}")
                can_run = False

            if not t.gpu and self.num_jobs_queue + t.num_jobs >= self.max_jobs_in_queue:
                self.logger.warning(f"Cant submit {t} because NUM_JOBS {t.num_jobs} would exceed {self.num_jobs_queue}/{self.max_jobs_in_queue}")
                can_run = False

            if can_run:
                return t
        return None

    def fail_task(self, t, running, failed, blocked):
        if t in running:
            running.remove(t)
        failed.append(t)
        self.logger.error(f"FAILED: {t}")
        if os.path.exists(t.hash_file):
            os.remove(t.hash_file)

        modified = True
        while modified:
            modified = False
            for t2 in self.tasks:
                for d in t2.dependencies:
                    if d in failed or d in blocked:
                        self.tasks.remove(t2)
                        blocked.append(t2)
                        modified = True
                        break

    def log_status(self, waiting, running, done, failed, blocked):
        self.logger.debug("")
        self.logger.debug(f"Status as of {time.ctime()}:")
        self.logger.debug(f"    Waiting: {[t.name for t in waiting]}")
        self.logger.debug(f"    Running: {[str(t) for t in running]}")
        if done:
            self.logger.debug(f"    Done:    {[t.name for t in done]}")
        if failed:
            self.logger.debug(f"    Failed:  {[t.name for t in failed]}")
        if blocked:
            self.logger.debug(f"    Blocked: {[t.name for t in blocked]}")
        self.logger.debug("")

        self.print_dashboard(waiting, running, done, failed, blocked)

    def get_subtasks(self, task_class, all_tasks):
        return [x for x in all_tasks if isinstance(x, task_class)]

    def get_string_with_colour(self, string, status):
        colours = {
            "waiting": Fore.BLACK + Style.BRIGHT,
            "running": Fore.WHITE + Style.BRIGHT,
            "done": Fore.GREEN + Style.BRIGHT,
            "failed": Fore.RED + Style.BRIGHT,
            "blocked": Fore.YELLOW + Style.DIM,
        }
        return colours[status] + string + Style.RESET_ALL

    def get_task_dashboard(self, task, waiting, running, done, failed, blocked):
        status = None
        if task in waiting:
            status = "waiting"
        elif task in running:
            status = "running"
        elif task in done:
            status = "done"
        elif task in failed:
            status = "failed"
        elif task in blocked:
            status = "blocked"
        return self.get_string_with_colour(task.name, status)

    def get_dashboard_line(self, stage, tasks, waiting, running, done, failed, blocked):
        strings = [self.get_task_dashboard(task, waiting, running, done, failed, blocked) for task in tasks]
        line_width = 160

        output = f"{stage:12s}"
        for i, s in enumerate(strings):
            if len(output + s) > line_width and i != len(strings) + 1:
                self.logger.info(output)
                output = " " * 12
            output += s + "   "
        self.logger.info(output)

        return output

    def print_dashboard(self, waiting, running, done, failed, blocked):
        all_tasks = waiting + running + done + failed + blocked

        self.logger.info("-------------------")
        self.logger.info("CURRENT TASK STATUS")

        options = ["WAITING", "RUNNING", "DONE", "FAILED", "BLOCKED"]
        header = "Key: " + "  ".join([self.get_string_with_colour(o, o.lower()) for o in options])
        self.logger.info(header)
        for name, task_class in zip(Manager.stages, Manager.task_order):
            tasks = self.get_subtasks(task_class, all_tasks)
            if tasks:
                self.get_dashboard_line(name, tasks, waiting, running, done, failed, blocked)

        self.logger.info("-------------------")

    def execute(self, check_config):
        self.logger.info(f"Executing pipeline for prefix {self.prefix}")
        self.logger.info(f"Output will be located in {self.output_dir}")
        if check_config:
            self.logger.info("Only verifying config, not launching anything")

        mkdirs(self.output_dir)
        c = self.run_config

        self.tasks = self.get_tasks(c)

        self.num_jobs_queue = 0
        self.num_jobs_queue_gpu = 0
        running_tasks = []
        done_tasks = []
        failed_tasks = []
        blocked_tasks = []
        squeue = None

        if check_config:
            self.logger.notice("Config verified, exiting")
            return

        self.print_dashboard(self.tasks, running_tasks, done_tasks, failed_tasks, blocked_tasks)

        start_sleep_time = self.global_config["OUTPUT"]["ping_frequency"]
        max_sleep_time = self.global_config["OUTPUT"]["max_ping_frequency"]
        current_sleep_time = start_sleep_time

        config_file_output = os.path.join(self.output_dir, os.path.basename(self.filename_path))
        if not check_config and self.filename_path != config_file_output:
            self.logger.info(f"Saving parsed config file from {self.filename_path} to {config_file_output}")
            shutil.copy(self.filename_path, config_file_output)
            chown_file(config_file_output)

        # Welcome to the primary loop
        while self.tasks or running_tasks:
            small_wait = False

            # Check status of current jobs
            for t in running_tasks:
                try:
                    completed = self.check_task_completion(t, blocked_tasks, done_tasks, failed_tasks, running_tasks, squeue)
                    small_wait = small_wait or completed
                except Exception as e:
                    self.logger.exception(e, exc_info=True)
                    self.fail_task(t, running_tasks, failed_tasks, blocked_tasks)

            # Submit new jobs if needed
            while self.num_jobs_queue < self.max_jobs:

                t = self.get_task_to_run(self.tasks, done_tasks)
                if t is not None:
                    self.logger.info("")
                    self.tasks.remove(t)
                    self.logger.notice(f"LAUNCHING: {t}")
                    try:
                        t.set_force_refresh(self.get_force_refresh(t))
                        t.set_force_ignore(self.get_force_ignore(t))
                        t.set_sbatch_cpu_header(self.sbatch_cpu_header)
                        t.set_sbatch_gpu_header(self.sbatch_gpu_header)
                        t.set_setup(self.task_setup)
                        started = t.run()
                    except Exception as e:
                        self.logger.exception(e, exc_info=True)
                        started = False
                    if started:
                        if t.gpu:
                            self.num_jobs_queue_gpu += t.num_jobs
                            message = (
                                f"LAUNCHED: {t} with {t.num_jobs} GPU NUM_JOBS. Total GPU NUM_JOBS now {self.num_jobs_queue_gpu}/{self.max_jobs_in_queue_gpu}"
                            )
                        else:
                            self.num_jobs_queue += t.num_jobs
                            message = f"LAUNCHED: {t} with {t.num_jobs} NUM_JOBS. Total NUM_JOBS now {self.num_jobs_queue}/{self.max_jobs_in_queue}"
                        self.logger.notice(message)
                        running_tasks.append(t)
                        completed = False
                        try:
                            completed = self.check_task_completion(t, blocked_tasks, done_tasks, failed_tasks, running_tasks, squeue)
                        except Exception as e:
                            self.logger.exception(e, exc_info=True)
                            self.fail_task(t, running_tasks, failed_tasks, blocked_tasks)
                        small_wait = small_wait or completed
                    else:
                        self.logger.error(f"FAILED TO LAUNCH: {t}")
                        self.fail_task(t, running_tasks, failed_tasks, blocked_tasks)
                    small_wait = True
                else:
                    break

            # Check quickly if we've added a new job, etc, in case of immediate failure
            if small_wait:
                self.log_status(self.tasks, running_tasks, done_tasks, failed_tasks, blocked_tasks)
                current_sleep_time = start_sleep_time
                time.sleep(0.1)
                squeue = None
            else:
                time.sleep(current_sleep_time)
                current_sleep_time *= 2
                if current_sleep_time > max_sleep_time:
                    current_sleep_time = max_sleep_time
                squeue = [i.strip() for i in subprocess.check_output(f"squeue -h -u $USER -o '%.200j'", shell=True, text=True).splitlines()]
                n = len(squeue)
                if n == 0 or n > self.max_jobs:
                    self.logger.debug(f"Squeue is reporting {n} NUM_JOBS in the queue... this is either 0 or toeing the line as to too many")

        self.log_finals(done_tasks, failed_tasks, blocked_tasks)

    def check_task_completion(self, t, blocked_tasks, done_tasks, failed_tasks, running_tasks, squeue):
        result = t.check_completion(squeue)
        # If its finished, good or bad, juggle tasks
        if result in [Task.FINISHED_SUCCESS, Task.FINISHED_FAILURE]:
            if t.gpu:
                self.num_jobs_queue_gpu -= t.num_jobs
            else:
                self.num_jobs_queue -= t.num_jobs
            if result == Task.FINISHED_SUCCESS:
                running_tasks.remove(t)
                self.logger.notice(f"FINISHED: {t} with {t.num_jobs} NUM_JOBS. NUM_JOBS now {self.num_jobs_queue}")
                done_tasks.append(t)
            else:
                self.fail_task(t, running_tasks, failed_tasks, blocked_tasks)
            chown_dir(t.output_dir)
            return True
        return False

    def log_finals(self, done_tasks, failed_tasks, blocked_tasks):
        self.logger.info("")
        self.logger.info("All tasks finished. Task summary as follows.")

        ws = self.message_store.get_warnings()
        es = self.message_store.get_errors()

        self.logger.info("Successfully completed tasks:")
        for t in done_tasks:
            self.logger.notice(f"\t{t}")
        if not done_tasks:
            self.logger.info("\tNo successful tasks")
        self.logger.info("Failed Tasks:")
        for t in failed_tasks:
            self.logger.error(f"\t{t}")
        if not failed_tasks:
            self.logger.info("\tNo failed tasks")
        self.logger.info("Blocked Tasks:")
        for t in blocked_tasks:
            self.logger.warning(f"\t{t}")
        if not blocked_tasks:
            self.logger.info("\tNo blocked tasks")

        self.logger.info("")
        if len(ws) == 0:
            self.logger.info(f"No warnings")
        else:
            self.logger.warning(f"{len(ws)} warnings")
        for w in ws:
            self.logger.warning(f"\t{w.message}")

        if len(es) == 0:
            self.logger.info(f"No errors")
        else:
            self.logger.error(f"{len(es)} errors")

        for w in es:
            self.logger.error(f"\t{w.message}")
