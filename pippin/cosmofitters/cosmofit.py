from pippin.task import Task


class CosmoFit(Task):
    """

    CONFIGURATION:
    ==============
    COSMOFIT:
        NAME_OF_FITTER (COSMOMC, WFIT, etc...)
            label:
                MASK: mask # partial match
                OPTS:
                    CHANGES_FOR_INDIVIDUAL_FITTERS

    OUTPUTS:
    ======
    """

    def get_tasks(
        task_config, prior_tasks, output_dir, stage_num, prefix, global_config
    ):
        from pippin.cosmofitters.factory import FitterFactory

        Task.logger.debug("Setting up CosmoFit tasks")

        tasks = []

        for fitter_name in task_config.get("COSMOFIT", []):
            Task.logger.info(f"Found fitter of type {fitter_name}, generating tasks.")
            config = {fitter_name: task_config["COSMOFIT"][fitter_name]}
            Task.logger.debug(f"Config for {fitter_name}: {config}")
            fitter = FitterFactory.get(fitter_name.lower())
            Task.logger.debug(f"Fitter class for {fitter_name}: {fitter}")
            if fitter is None:
                Task.logger.error(
                    f"Fitter of type {fitter_name} not found, perhaps it's a typo? Skipping."
                )
                continue
            Task.logger.debug(
                f"get_task function for {fitter_name}: {fitter.get_tasks}"
            )
            ts = fitter.get_tasks(
                config, prior_tasks, output_dir, stage_num, prefix, global_config
            )
            Task.logger.debug(f"{fitter} tasks: {ts}")
            tasks += ts
            if len(tasks) == 0:
                Task.fail_config("No CosmoFit tasks generated!")
            Task.logger.info(f"Generated {len(tasks)} CosmoFit tasks.")
        return tasks
