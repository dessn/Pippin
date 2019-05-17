import os
import inspect
import subprocess
import time

from pippin.aggregator import Aggregator
from pippin.classifiers.classifier import Classifier
from pippin.classifiers.factory import ClassifierFactory
from pippin.config import get_logger, get_config
from pippin.dataprep import DataPrep
from pippin.merge import Merger
from pippin.snana_fit import SNANALightCurveFit
from pippin.snana_sim import SNANASimulation
from pippin.task import Task


class Manager:
    stages = {
        "DATAPREP": 0,
        "SIM": 1,
        "LCFIT": 2,
        "CLASSIFY": 3,
        "AGGREGATE": 4,
        "MERGE": 5,
        "BIASCOR": 6
    }

    def __init__(self, filename, config, message_store):
        self.logger = get_logger()
        self.message_store = message_store
        self.filename = filename
        self.run_config = config
        self.global_config = get_config()

        self.prefix = self.global_config["GLOBAL"]["prefix"] + "_" + filename
        self.max_jobs = int(self.global_config["GLOBAL"]["max_jobs"])
        self.max_jobs_in_queue = int(self.global_config["GLOBAL"]["max_jobs_in_queue"])

        self.output_dir = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/../" + self.global_config['OUTPUT']['output_dir'] + "/" + self.filename)
        self.tasks = None

        self.start = None
        self.finish = None
        self.force_refresh = False

    def get_force_refresh(self, task):
        if self.start is None:
            return self.force_refresh
        return task.stage >= self.start

    def set_force_refresh(self, force_refresh):
        self.force_refresh = force_refresh

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
            assert key in Manager.stages.keys(), f"Stage {key} is not in recognised keys {Manager.stages.keys()}"
            num = Manager.stages[key]
        assert num in Manager.stages.values(), f"Stage {num} is not in recognised values {Manager.stages.values()}"
        return num

    def get_tasks(self, config):
        data_tasks = self.get_dataset_prep_tasks(config)
        sim_tasks = self.get_simulation_tasks(config)
        lcfit_tasks = self.get_lcfit_tasks(config, sim_tasks + data_tasks)
        classification_tasks = self.get_classification_tasks(config, sim_tasks + data_tasks, lcfit_tasks)
        aggregator_tasks = self.get_aggregator_tasks(config, classification_tasks)
        merger_tasks = self.get_merge_tasks(config, aggregator_tasks, lcfit_tasks)
        total_tasks = data_tasks + sim_tasks + lcfit_tasks + classification_tasks + aggregator_tasks + merger_tasks
        self.logger.info("")
        self.logger.notice("Listing tasks:")
        for task in total_tasks:
            self.logger.notice(f"\t{task}")
        self.logger.info("")
        return total_tasks

    def get_dataset_prep_tasks(self, c):
        tasks = []
        stage = Manager.stages["DATAPREP"]
        if self.finish is not None and self.finish <= stage:
            return tasks
        for name in c.get("DATAPREP", []):
            output_dir = self._get_data_prep_output_dir(name)
            s = DataPrep(name, output_dir, c["DATAPREP"][name]["OPTS"])
            s.set_stage(stage)
            self.logger.debug(f"Creating data prep task {name} with {s.num_jobs} jobs, output to {output_dir}")
            tasks.append(s)
        return tasks

    def get_simulation_tasks(self, c):
        tasks = []
        stage = Manager.stages["SIM"]
        if self.finish is not None and self.finish <= stage:
            return tasks
        for sim_name in c.get("SIM", []):
            sim_output_dir = self._get_sim_output_dir(sim_name)
            s = SNANASimulation(sim_name, sim_output_dir, f"{self.prefix}_{sim_name}", c["SIM"][sim_name], self.global_config)
            s.set_stage(stage)
            self.logger.debug(f"Creating simulation task {sim_name} with {s.num_jobs} jobs, output to {sim_output_dir}")
            tasks.append(s)
        return tasks

    def get_lcfit_tasks(self, c, sim_tasks):
        tasks = []
        stage = Manager.stages["LCFIT"]
        if self.finish is not None and self.finish <= stage:
            return tasks
        for fit_name in c.get("LCFIT", []):
            fit_config = c["LCFIT"][fit_name]
            for sim in sim_tasks:
                if fit_config.get("MASK") is None or fit_config.get("MASK") in sim.name:
                    fit_output_dir = self._get_lc_output_dir(sim.name, fit_name)
                    f = SNANALightCurveFit(fit_name, fit_output_dir, sim, fit_config, self.global_config)
                    f.set_stage(stage)
                    self.logger.info(f"Creating fitting task {fit_name} with {f.num_jobs} jobs, for simulation {sim.name}")
                    tasks.append(f)
        return tasks

    def get_classification_tasks(self, c, sim_tasks, lcfit_tasks):
        tasks = []
        stage = Manager.stages["CLASSIFY"]
        if self.finish is not None and self.finish <= stage:
            return tasks
        for clas_name in c.get("CLASSIFICATION", []):
            config = c["CLASSIFICATION"][clas_name]
            name = config["CLASSIFIER"]
            cls = ClassifierFactory.get(name)
            options = config.get("OPTS", {})
            mode = config["MODE"].lower()
            assert mode in ["train", "predict"], "MODE should be either train or predict"
            if mode == "train":
                mode = Classifier.TRAIN
            else:
                mode = Classifier.PREDICT

            needs_sim, needs_lc = cls.get_requirements(options)

            runs = []
            if needs_sim and needs_lc:
                runs = [(l.dependencies[0], l) for l in lcfit_tasks]
            elif needs_sim:
                runs = [(s, None) for s in sim_tasks]
            elif needs_lc:
                runs = [(l.dependencies[0], l) for l in lcfit_tasks]
            else:
                self.logger.warn(f"Classifier {name} does not need sims or fits. Wat.")

            num_gen = 0
            mask = config.get("MASK", "")
            mask_sim = config.get("MASK_SIM", "")
            mask_fit = config.get("MASK_FIT", "")
            for s, l in runs:
                sim_name = s.name if s is not None else None
                fit_name = l.name if l is not None else None
                if sim_name is not None and (mask not in sim_name or mask_sim not in sim_name):
                    continue
                if fit_name is not None and (mask not in fit_name or mask_fit not in fit_name):
                    continue
                deps = []
                if s is not None:
                    deps.append(s)
                if l is not None:
                    deps.append(l)

                model = options.get("MODEL")
                if model is not None:
                    for t in tasks:
                        if model == t.name:
                            deps.append(t)

                clas_output_dir = self._get_clas_output_dir(sim_name, fit_name, clas_name)
                cc = cls(clas_name, clas_output_dir, deps, mode, options)
                cc.set_stage(stage)
                self.logger.info(f"Creating classification task {name} with {cc.num_jobs} jobs, for LC fit {fit_name} on simulation {sim_name}")
                num_gen += 1
                tasks.append(cc)
            if num_gen == 0:
                self.logger.error(f"Classifier {name} with mask {mask} matched no combination of sims and fits")
        return tasks

    def get_aggregator_tasks(self, c, classifier_tasks):
        tasks = []
        stage = Manager.stages["AGGREGATE"]
        if self.finish is not None and self.finish < Manager.stages["AGGREGATE"]:
            return tasks
        for agg_name in c.get("AGGREGATION", []):
            config = c["AGGREGATION"][agg_name]
            options = config.get("OPTS", {})
            mask = config.get("MASK")
            deps = [c for c in classifier_tasks if mask is None or mask in c.name]
            if len(deps) == 0:
                self.logger.error("Aggregator {agg_name} with mask {mask} matched no classifier tasks")
            else:
                a = Aggregator(agg_name, self._get_aggregator_dir(agg_name), deps, options)
                a.set_stage(stage)
                self.logger.info(f"Creating aggregation task {agg_name} with {a.num_jobs}")
                tasks.append(a)
        return tasks

    def get_merge_tasks(self, c, agg_tasks, lcfit_tasks):
        tasks = []
        stage = Manager.stages["MERGE"]
        if self.finish is not None and self.finish < Manager.stages["MERGE"]:
            return tasks
        for name in c.get("MERGE", []):
            num_gen = 0
            config = c["MERGE"][name]
            options = config.get("OPTS", {})
            mask = config.get("MASK", "")
            mask_sim = config.get("MASK_SIM", "")
            mask_lc = config.get("MASK_FIT", "")
            mask_agg = config.get("MASK_AGG", "")

            for lcfit in lcfit_tasks:
                if mask and mask not in lcfit.name:
                    continue
                if mask_lc and mask_lc not in lcfit.name:
                    continue
                sim = lcfit.get_dep(SNANASimulation)
                if mask and mask not in sim.name:
                    continue
                if mask_sim and mask_sim not in sim.name:
                    continue

                for agg in agg_tasks:
                    if mask_agg and mask_agg not in agg.name:
                        continue
                    if mask and mask not in agg.name:
                        continue
                    num_gen += 1
                    task = Merger(name, self._get_merge_output_dir(name, lcfit.name, agg.name), [lcfit, agg], options)
                    task.set_stage(stage)
                    self.logger.info(f"Creating aggregation task {name} with {task.num_jobs}")
                    tasks.append(task)
            if num_gen == 0:
                self.logger.error(f"Merger {name} with mask {mask} matched no combination of aggregators and fits")
        return tasks

    def get_num_running_jobs(self):
        num_jobs = int(subprocess.check_output("squeue -ho %A -u $USER | wc -l", shell=True, stderr=subprocess.STDOUT))
        return num_jobs

    def get_task_to_run(self, tasks_to_run, done_tasks):
        for t in tasks_to_run:
            can_run = True
            for dep in t.dependencies:
                if dep not in done_tasks:
                    can_run = False
            if can_run:
                return t
        return None

    def fail_task(self, t, running, failed, blocked):
        if t in running:
            running.remove(t)
        failed.append(t)
        self.logger.error(f"Task {t} failed")
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
        self.logger.debug(f"    Running: {[t.name for t in running]}")
        if done:
            self.logger.debug(f"    Done:    {[t.name for t in done]}")
        if failed:
            self.logger.debug(f"    Failed:  {[t.name for t in failed]}")
        if blocked:
            self.logger.debug(f"    Blocked: {[t.name for t in blocked]}")
        self.logger.debug("")

    def execute(self):
        self.logger.info(f"Executing pipeline for prefix {self.prefix}")
        self.logger.info(f"Output will be located in {self.output_dir}")
        c = self.run_config

        self.tasks = self.get_tasks(c)
        running_tasks = []
        done_tasks = []
        failed_tasks = []
        blocked_tasks = []
        squeue = []

        # Welcome to the primary loop
        while self.tasks or running_tasks:
            small_wait = False

            # Check status of current jobs
            for t in running_tasks:
                result = t.check_completion(squeue)
                # If its finished, good or bad, juggle tasks
                if result in [Task.FINISHED_SUCCESS, Task.FINISHED_FAILURE]:
                    if result == Task.FINISHED_SUCCESS:
                        running_tasks.remove(t)
                        self.logger.notice(f"Task {t} finished successfully")
                        done_tasks.append(t)
                    else:
                        self.fail_task(t, running_tasks, failed_tasks, blocked_tasks)
                    small_wait = True

            # Submit new jobs if needed
            num_running = self.get_num_running_jobs()
            while num_running < self.max_jobs:
                t = self.get_task_to_run(self.tasks, done_tasks)
                if t is not None:
                    self.logger.info("")
                    self.tasks.remove(t)
                    self.logger.notice(f"LAUNCHING: {t}")
                    started = t.run(self.get_force_refresh(t))
                    if started:
                        num_running += t.num_jobs
                        self.logger.notice(f"RUNNING: {t}")
                        running_tasks.append(t)
                    else:
                        self.logger.error(f"FAILED TO LAUNCH: {t}")
                        self.fail_task(t, running_tasks, failed_tasks, blocked_tasks)
                    small_wait = True
                else:
                    break

            # Check quickly if we've added a new job, etc, in case of immediate failure
            if small_wait:
                self.log_status(self.tasks, running_tasks, done_tasks, failed_tasks, blocked_tasks)
                time.sleep(0.5)
            else:
                time.sleep(self.global_config["OUTPUT"].getint("ping_frequency"))
                squeue = subprocess.check_output(f"squeue -h -u $USER -o '%.70j'", shell=True).split("\n")
                print(squeue)

        self.log_finals(done_tasks, failed_tasks, blocked_tasks)

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

    def _get_data_prep_output_dir(self, name):
        return f"{self.output_dir}/{Manager.stages['DATAPREP']}_DATAPREP/{name}"

    def _get_sim_output_dir(self, sim_name):
        return f"{self.output_dir}/{Manager.stages['SIM']}_SIM/{sim_name}"

    def _get_phot_output_dir(self, sim_name):
        return f"{self.output_dir}/{Manager.stages['SIM']}_SIM/{sim_name}/{self.prefix}_{sim_name}"

    def _get_lc_output_dir(self, sim_name, fit_name):
        return f"{self.output_dir}/{Manager.stages['LCFIT']}_LCFIT/{fit_name}_{sim_name}"

    def _get_clas_output_dir(self, sim_name, fit_name, clas_name):
        fit_name = "" if fit_name is None else "_" + fit_name
        sim_name = "" if sim_name is None else "_" + sim_name
        return f"{self.output_dir}/{Manager.stages['CLASSIFY']}_CLAS/{clas_name}{sim_name}{fit_name}"

    def _get_aggregator_dir(self, agg_name):
        return f"{self.output_dir}/{Manager.stages['AGGREGATE']}_AGG/{agg_name}"

    def _get_merge_output_dir(self, merge_name, lcfit_name, agg_name):
        return f"{self.output_dir}/{Manager.stages['MERGE']}_MERGE/{merge_name}_{lcfit_name}_{agg_name}"


if __name__ == "__main__":
    import logging
    import yaml
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)8s |%(filename)20s:%(lineno)3d |%(funcName)25s]   %(message)s")

    with open("../configs/test.yml", "r") as f:
        cc = yaml.safe_load(f)

    manager = Manager("test", cc)
    manager.execute()
