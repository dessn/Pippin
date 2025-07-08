# ============
#
#  History
#
# Jul 7 2025: R.Kessler :
#   To identify SNIa model, replace SALT check with more general list that
#   includes BAYESN; see global SUBSTRING_SNIA_LIST

import os
import shutil
import subprocess
import tempfile
import json

from pippin.base import ConfigBasedExecutable
from pippin.config import (
    chown_dir,
    copytree,
    mkdirs,
    get_data_loc,
    get_hash,
    read_yaml,
    get_config,
)
from pippin.task import Task

# define substring list to identify SNIa model
SUBSTRING_SNIA_LIST = [ 'SALT',  'BAYESN', 'bayesn', 
                        'SNEMO', 'BYOSED', 'mlcs', 'snoopy' ]  # R.Kessler July 7 2025



class SNANASimulation(ConfigBasedExecutable):
    """Merge fitres files and aggregator output

    CONFIGURATION:
    ==============
    SIM:
      label:
        IA_name:  # Must have a sim type that starts with IA_ or is called exactly IA. Its how it gets divided into Ia or contaminant.
          BASE: sn_ia_salt2_g10_des3yr.input # Location of an input file. Either in the data_files dir or full path
          DNDZ_ALLSCALE: 3.0  # Override input file content here
        II_JONES:
          BASE: sn_collection_jones.input
        GLOBAL:
            NGEN_UNIT: 1  # Set properties for all input file shere


    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        genversion: genversion of sim
        types_dict: dict map from IA or NONIA to numeric gentypes
        types: dict map from numeric gentype to string (Ia, II, etc)
        photometry_dirs: location of fits files with photometry. is a list.
        ranseed_change: true or false for if RANSEED_CHANGE was set
        ranseed_change_val: the value of ranseed change (None if it wasnt set)
        blind: bool - whether to blind cosmo results
    """

    def __init__(
        self, name, output_dir, config, global_config, combine="combine.input"
    ):
        self.data_dirs = global_config["DATA_DIRS"]
        base_file = get_data_loc(combine)
        super().__init__(name, output_dir, config, base_file, ": ")

        # Check for any replacements
        path_sndata_sim = get_config().get("SNANA").get("sim_dir")
        self.logger.debug(f"Setting PATH_SNDATA_SIM to {path_sndata_sim}")
        self.yaml["CONFIG"]["PATH_SNDATA_SIM"] = path_sndata_sim

        self.genversion = self.config["GENVERSION"]
        if len(self.genversion) < 30:
            self.genprefix = self.genversion
        else:
            hash = get_hash(self.genversion)[:5]
            self.genprefix = self.genversion[:25] + hash

        self.options = self.config.get("OPTS", {})

        self.reserved_keywords = ["BASE"]
        self.reserved_top = ["GENVERSION", "GLOBAL", "OPTS", "EXTERNAL"]
        self.config_path = f"{self.output_dir}/{self.genversion}.input"  # Make sure this syncs with the tmp file name
        self.global_config = global_config

        self.batch_replace = self.options.get(
            "BATCH_REPLACE", self.global_config.get("BATCH_REPLACE", {})
        )
        batch_mem = self.batch_replace.get("REPLACE_MEM", None)
        if batch_mem is not None:
            self.yaml["CONFIG"]["BATCH_MEM"] = batch_mem
        batch_walltime = self.batch_replace.get("REPLACE_WALLTIME", None)
        if batch_walltime is not None:
            self.yaml["CONFIG"]["BATCH_WALLTIME"] = batch_walltime

        self.sim_log_dir = f"{self.output_dir}/LOGS"
        self.total_summary = os.path.join(self.sim_log_dir, "MERGE.LOG")
        self.done_file = f"{self.output_dir}/LOGS/ALL.DONE"
        self.logging_file = self.config_path.replace(".input", ".LOG")
        self.kill_file = self.config_path.replace(".input", "_KILL.LOG")

        if "EXTERNAL" not in self.config.keys():
            # Deterime the type of each component
            keys = [k for k in self.config.keys() if k not in self.reserved_top]
            self.base_ia = []
            self.base_cc = []
            types = {}
            types_dict = {"IA": [], "NONIA": []}
            for k in keys:
                d = self.config[k]
                base_file = d.get("BASE")
                if base_file is None:
                    Task.fail_config(
                        f"Your simulation component {k} for sim name {self.name} needs to specify a BASE input file"
                    )
                base_path = get_data_loc(base_file)
                if base_path is None:
                    Task.fail_config(
                        f"Cannot find sim component {k} base file at {base_path} for sim name {self.name}"
                    )

                gentype, genmodel = None, None
                with open(base_path) as f:
                    for line in f.read().splitlines():
                        if line.upper().strip().startswith("GENTYPE:"):
                            gentype = line.upper().split(":")[1].strip()
                        if line.upper().strip().startswith("GENMODEL:"):
                            genmodel = line.upper().split(":")[1].strip()

                gentype = gentype or d.get("GENTYPE")
                if gentype is None:
                    self.fail_config(
                        f"The simulation component {k} needs to specify a GENTYPE in its input file"
                    )
                gentype = int(gentype)
                genmodel = genmodel or d.get("GENMODEL")

                if not gentype:
                    gentype_sublist = self.get_simInput_key_values(
                        base_path, ["GENTYPE"]
                    )
                    self.logger.debug(f"gentype_sublist: {gentype_sublist}")
                    if len(gentype_sublist) == 0:
                        Task.fail_config(
                            f"Cannot find GENTYPE for component {k} and base file {base_path}, or any included files"
                        )
                    else:
                        gentype = gentype_sublist[0]
                if not genmodel:
                    genmodel_sublist = self.get_simInput_key_values(
                        base_path, ["GENMODEL"]
                    )["GENMODEL"]
                    self.logger.debug(f"genmodel_sublist: {genmodel_sublist}")
                    if len(genmodel_sublist) == 0:
                        Task.fail_config(
                            f"Cannot find GENMODEL for component {k} and base file {base_path}, or any included files"
                        )
                    else:
                        genmodel = genmodel_sublist[0]

                type2 = 100 + gentype
                is_snia = any(substring in genmodel for substring in SUBSTRING_SNIA_LIST) # RK, Jul 7 2025
                # xxx mark delete if "SALT" in genmodel:
                if is_snia:
                    self.base_ia.append(base_file)
                    types[gentype] = "Ia"
                    types[type2] = "Ia"
                    types_dict["IA"].append(gentype)
                    types_dict["IA"].append(type2)
                else:
                    self.base_cc.append(base_file)
                    types[gentype] = "II"
                    types[type2] = "II"
                    types_dict["NONIA"].append(gentype)
                    types_dict["NONIA"].append(type2)

            sorted_types = dict(sorted(types.items()))
            self.logger.debug(f"Types found: {json.dumps(sorted_types)}")
            self.output["types_dict"] = types_dict
            self.output["types"] = sorted_types

            rankeys = [
                r
                for r in self.config.get("GLOBAL", {}).keys()
                if r.startswith("RANSEED_")
            ]
            value = (
                int(self.config["GLOBAL"][rankeys[0]].split(" ")[0]) if rankeys else 1
            )
            self.set_num_jobs(2 * value)

            self.output["blind"] = self.options.get("BLIND", False)
            self.derived_batch_info = None

            # Determine if all the top level input files exist
            if len(self.base_ia + self.base_cc) == 0:
                Task.fail_config(
                    "Your sim has no components specified! Please add something to simulate!"
                )

            # Try to determine how many jobs will be put in the queue
            # First see if it's been explicitly set
            num_jobs = self.options.get("NUM_JOBS")
            if num_jobs is not None:
                self.num_jobs = num_jobs
                self.logger.debug(f"Num jobs set by NUM_JOBS option to {self.num_jobs}")
            else:
                try:
                    # If BATCH_INFO is set, we'll use that
                    batch_info = self.config.get("GLOBAL", {}).get("BATCH_INFO")
                    default_batch_info = self.yaml["CONFIG"].get("BATCH_INFO")

                    # If its not set, lets check for ranseed_repeat or ranseed_change
                    if batch_info is None:
                        ranseed_repeat = self.config.get("GLOBAL", {}).get(
                            "RANSEED_REPEAT"
                        )
                        ranseed_change = self.config.get("GLOBAL", {}).get(
                            "RANSEED_CHANGE"
                        )
                        default = self.yaml.get("CONFIG", {}).get("RANSEED_REPEAT")
                        ranseed = ranseed_repeat or ranseed_change or default

                        if ranseed:
                            num_jobs = int(ranseed.strip().split()[0])
                            self.logger.debug(
                                f"Found a randseed with {num_jobs}, deriving batch info"
                            )
                            comps = default_batch_info.strip().split()
                            comps[-1] = str(num_jobs)
                            self.derived_batch_info = " ".join(comps)
                            self.num_jobs = num_jobs
                            self.logger.debug(
                                f"Num jobs set by RANSEED to {self.num_jobs}"
                            )
                    else:
                        # self.logger.debug(f"BATCH INFO property detected as {property}")
                        self.num_jobs = int(batch_info.split()[-1])
                        self.logger.debug(
                            f"Num jobs set by BATCH_INFO to {self.num_jobs}"
                        )
                except Exception:
                    self.logger.warning(
                        f"Unable to determine how many jobs simulation {self.name} has"
                    )
                    self.num_jobs = 1

            self.output["genversion"] = self.genversion
            self.output["genprefix"] = self.genprefix

            self.ranseed_change = self.config.get("GLOBAL", {}).get("RANSEED_CHANGE")
            base = os.path.expandvars(self.global_config["SNANA"]["sim_dir"])
            self.output["ranseed_change"] = self.ranseed_change is not None
            self.output["ranseed_change_val"] = self.ranseed_change
            self.get_sim_folders(base, self.genversion)
            self.output["sim_folders"] = self.sim_folders
        else:
            self.sim_folders = self.output["sim_folders"]

    def get_sim_folders(self, base, genversion):
        if self.output.get("ranseed_change"):
            num_sims = int(self.output["ranseed_change_val"].split()[0])
            self.logger.debug(
                f"Detected randseed change with {num_sims} sims, updating sim_folders"
            )
            self.sim_folders = [
                os.path.join(base, genversion) + f"-{i + 1:04d}"
                for i in range(num_sims)
            ]
            self.logger.debug(f"First sim folder set to {self.sim_folders[0]}")

        else:
            self.sim_folders = [os.path.join(base, genversion)]

    def write_input(self):
        # As Pippin only does one GENVERSION at a time, lets extract it first, and also the config
        c = self.yaml["CONFIG"]
        d = self.yaml["GENVERSION_LIST"][0]
        g = self.yaml["GENOPT_GLOBAL"]

        # Ensure g is a dict with a ref we can update
        if g is None:
            g = {}
            self.yaml["GENOPT_GLOBAL"] = g

        # Start setting properties in the right area
        d["GENVERSION"] = self.genversion

        # Logging now goes in the "CONFIG"
        c["LOGDIR"] = os.path.basename(self.sim_log_dir)

        for k in self.config.keys():
            if k.upper() not in self.reserved_top:
                run_config = self.config[k]
                run_config_keys = list(run_config.keys())
                assert "BASE" in run_config_keys, (
                    "You must specify a base file for each option"
                )
                for key in run_config_keys:
                    if key.upper() in self.reserved_keywords:
                        continue
                    base_file = run_config["BASE"]
                    match = os.path.basename(base_file).split(".")[0]
                    val = run_config[key]
                    if not isinstance(val, list):
                        val = [val]

                    lookup = f"GENOPT({match})"
                    if lookup not in d:
                        d[lookup] = {}
                    for v in val:
                        d[lookup][key] = v

        if len(self.data_dirs) > 1:
            data_dir = self.data_dirs[0]
            c["PATH_USER_INPUT"] = data_dir

        for key in self.config.get("GLOBAL", []):
            if key.upper() == "BASE":
                continue
            direct_set = [
                "FORMAT_MASK",
                "RANSEED_REPEAT",
                "RANSEED_CHANGE",
                "BATCH_INFO",
                "BATCH_MEM",
                "NGEN_UNIT",
                "RESET_CIDOFF",
            ]
            if key in direct_set:
                c[key] = self.config["GLOBAL"][key]
            else:
                g[key] = self.config["GLOBAL"][key]

            if self.derived_batch_info:
                c["BATCH_INFO"] = self.derived_batch_info

            if key == "RANSEED_CHANGE" and c.get("RANSEED_REPEAT") is not None:
                del c["RANSEED_REPEAT"]
            elif key == "RANSEED_REPEAT" and c.get("RANSEED_CHANGE") is not None:
                del c["RANSEED_CHANGE"]

        if self.base_ia:
            c["SIMGEN_INFILE_Ia"] = [os.path.basename(f) for f in self.base_ia]
        else:
            del c["SIMGEN_INFILE_Ia"]

        if self.base_cc:
            c["SIMGEN_INFILE_NONIa"] = [os.path.basename(f) for f in self.base_cc]
        else:
            del c["SIMGEN_INFILE_NONIa"]

        c["GENPREFIX"] = self.genprefix

        # Put config in a temp directory
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name

        # Copy the base files across
        input_paths = []
        for f in self.base_ia + self.base_cc:
            resolved = get_data_loc(f)
            shutil.copy(resolved, temp_dir)
            input_paths.append(os.path.join(temp_dir, os.path.basename(f)))
            self.logger.debug(f"Copying input file {resolved} to {temp_dir}")

        # Copy the include input file if there is one
        input_copied = []
        fs = self.base_ia + self.base_cc
        for ff in fs:
            if ff not in input_copied:
                input_copied.append(ff)
                path = get_data_loc(ff)
                copied_path = os.path.join(temp_dir, os.path.basename(path))
                with open(path, "r") as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line.startswith("INPUT_FILE_INCLUDE"):
                            include_file = line.split(":")[-1].strip()
                            include_file_path = get_data_loc(include_file)
                            self.logger.debug(
                                f"Copying INPUT_FILE_INCLUDE file {include_file_path} to {temp_dir}"
                            )

                            include_file_basename = os.path.basename(include_file_path)
                            include_file_output = os.path.join(
                                temp_dir, include_file_basename
                            )

                            if include_file_output not in input_copied:
                                # Copy include file into the temp dir
                                shutil.copy(include_file_path, temp_dir)

                                # Then SED the file to replace the full path with just the basename
                                if include_file != include_file_basename:
                                    sed_command = f"sed -i -e 's|{include_file}|{include_file_basename}|g' {copied_path}"
                                    self.logger.debug(
                                        f"Running sed command: {sed_command}"
                                    )
                                    subprocess.run(
                                        sed_command,
                                        stderr=subprocess.STDOUT,
                                        cwd=temp_dir,
                                        shell=True,
                                    )

                                # And make sure we dont do this file again
                                fs.append(include_file_output)

        # Write the primary input file
        main_input_file = f"{temp_dir}/{self.genversion}.input"
        self.write_output_file(main_input_file)

        # Remove any duplicates and order the output files
        output_files = [f"{temp_dir}/{a}" for a in sorted(os.listdir(temp_dir))]
        self.logger.debug(
            f"{len(output_files)} files used to create simulation. Hashing them."
        )

        # Get current hash
        new_hash = self.get_hash_from_files(output_files)
        regenerate = self._check_regenerate(new_hash)

        if regenerate:
            self.logger.info("Running simulation")
            # Clean output dir. God I feel dangerous doing this, so hopefully unnecessary check
            if "//" not in self.output_dir and len(self.output_dir) > 30:
                self.logger.debug(f"Cleaning output directory {self.output_dir}")
                shutil.rmtree(self.output_dir, ignore_errors=True)
                mkdirs(self.output_dir)
                self.logger.debug(f"Copying from {temp_dir} to {self.output_dir}")
                copytree(temp_dir, self.output_dir)
                self.save_new_hash(new_hash)
            else:
                self.logger.error(
                    f"Seems to be an issue with the output dir path: {self.output_dir}"
                )

            chown_dir(self.output_dir)
        else:
            self.logger.info("Hash check passed, not rerunning")
        temp_dir_obj.cleanup()
        return regenerate, new_hash

    def _run(self):
        regenerate, new_hash = self.write_input()
        if not regenerate:
            self.should_be_done()
            return True

        with open(self.logging_file, "w") as f:
            subprocess.run(
                ["submit_batch_jobs.sh", os.path.basename(self.config_path)],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=self.output_dir,
            )

        self.logger.info(f"Sim running and logging outputting to {self.logging_file}")
        return True

    def kill_and_fail(self):
        with open(self.kill_file, "w") as f:
            self.logger.info(f"Killing remaining jobs for {self.name}")
            subprocess.run(
                ["submit_batch_jobs.sh", "--kill", os.path.basename(self.config_path)],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=self.output_dir,
            )
        return Task.FINISHED_FAILURE

    def check_issues(self):
        log_files = [self.logging_file]
        if os.path.exists(self.sim_log_dir):
            log_files += [
                os.path.join(self.sim_log_dir, f)
                for f in os.listdir(self.sim_log_dir)
                if f.upper().endswith(".LOG")
            ]
        else:
            self.logger.warning(
                f"Warning, sim log dir {self.sim_log_dir} does not exist. Something might have gone terribly wrong"
            )
        self.scan_files_for_error(
            log_files,
            "FATAL ERROR ABORT",
            "QOSMaxSubmitJobPerUserLimit",
            "DUE TO TIME LIMIT",
        )
        return self.kill_and_fail()

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file) or not os.path.exists(self.total_summary):
            if os.path.exists(self.done_file):
                self.logger.info(f"Simulation {self.name} found done file!")
                with open(self.done_file) as f:
                    if "FAIL" in f.read():
                        self.logger.error(
                            f"Done file {self.done_file} reporting failure"
                        )
                        return self.check_issues()
            else:
                self.logger.error("MERGE.LOG was not created, job died on submission")
                return self.check_issues()

            if os.path.exists(self.total_summary):
                y = read_yaml(self.total_summary)
                if "MERGE" in y.keys():
                    for i, row in enumerate(y["MERGE"]):
                        if (
                            len(row) == 6
                        ):  # Old version for backward compatibility (before 15/01/2021)
                            state, iver, version, ngen, nwrite, cpu = row
                        else:  # New MERGE.LOG syntax (after 15/01/2021)
                            state, iver, version, ngen, nwrite, nspec, cpu = row
                        if cpu < 60:
                            units = "minutes"
                        else:
                            cpu = cpu / 60
                            units = "hours"
                        self.logger.info(
                            f"Simulation {i + 1} generated {ngen} events and wrote {nwrite} to file, taking {cpu:0.1f} CPU {units}"
                        )
                else:
                    self.logger.error(
                        f"File {self.total_summary} does not have a MERGE section - did it die?"
                    )
                    return self.kill_and_fail()
                if "SURVEY" in y.keys():
                    self.output["SURVEY"] = y["SURVEY"]
                    self.output["SURVEY_ID"] = y["IDSURVEY"]
                else:
                    self.output["SURVEY"] = "UNKNOWN"
                    self.output["SURVEY_ID"] = 0

            else:
                self.logger.warning(f"Cannot find {self.total_summary}")

            self.logger.info("Done file found, creating symlinks")
            s_ends = [
                os.path.join(self.output_dir, os.path.basename(s))
                for s in self.sim_folders
            ]
            for s, s_end in zip(self.sim_folders, s_ends):
                if not os.path.exists(s_end):
                    # Check to make sure there isn't a broken symlink at s_end
                    # os.path.exists will return false for broken symlinks, even if one exists
                    if os.path.islink(s_end):
                        self.logger.error(
                            f"Symlink {s_end} exists and is pointing to a broken or missing directory"
                        )
                        return Task.FINISHED_FAILURE
                    else:
                        self.logger.debug(f"Linking {s} -> {s_end}")
                        os.symlink(s, s_end, target_is_directory=True)
                chown_dir(self.output_dir)
            self.output.update({"photometry_dirs": s_ends})
            return Task.FINISHED_SUCCESS

        return self.check_for_job(squeue, f"{self.genversion}.input-CPU")

    def get_simInput_key_values(self, sim_input_file, key_list):
        # Created Feb 2024 by R.Kessler
        # Example:
        #    input key_list = [ 'GENMODEL', 'GENRANGE_PEAKMJD' ]
        # Search sim_input_file for these keys, and also search
        # INCLUDE files. Sim-input format is not quite YAML,
        # so there can be multiple values returned for a given key.
        # Return a dictionary as follows:
        #   { 'GENMODEL':  [ arg list ],
        #     'GENRANGE_PEAKMJD' : [arg list] }
        #
        # Initial use is for pippin to read GENMODEL keys, so here
        # is just a public place to store this utility.
        # .xyz

        INPUT_FILE_LIST = [sim_input_file]

        # first find INCLUDE files and append INPUT_FILE_LIST
        KEYLIST_INCLUDE_FILE = ["INPUT_INCLUDE_FILE", "INPUT_FILE_INCLUDE"]
        with open(sim_input_file, "rt") as f:
            line_list = f.readlines()
        for line in line_list:
            line = line.rstrip()  # remove trailine space
            wdlist = line.split()
            if len(wdlist) < 2:
                continue
            for key in KEYLIST_INCLUDE_FILE:
                if wdlist[0] == key + ":":
                    inc_file = os.path.expandvars(wdlist[1])
                    INPUT_FILE_LIST.append(inc_file)

        # - - - - -
        # init output dictionary
        key_dict = {}
        for key in key_list:
            key_dict[key] = []

        # print(f" xxx util: INPUT_FILE_LIST = {INPUT_FILE_LIST}")

        # loop over all of the sim-input files and search for key values
        for inp_file in INPUT_FILE_LIST:
            f = open(inp_file, "rt")
            line_list = f.readlines()
            f.close()

            # print(f" Inspect {inp_file}")
            for line in line_list:
                if line.isspace():
                    continue
                if line[0:1] == "#":
                    continue
                line = line.rstrip()  # remove trailing space
                line = line.split("#")[0]  # remove comments
                wdlist = line.split()
                if len(wdlist) < 2:
                    continue
                for key in key_list:
                    if wdlist[0] == key + ":":
                        args = " ".join(wdlist[1:])
                        # print(f" xxx load {key} with {args}")
                        key_dict[key].append(args)

        # - - - - -

        return key_dict
        # end get_simInput_key_values

    @staticmethod
    def get_tasks(
        config, prior_tasks, base_output_dir, stage_number, prefix, global_config
    ):
        tasks = []
        for sim_name in config.get("SIM", []):
            task_config = config["SIM"][sim_name]
            if "EXTERNAL" not in task_config.keys():
                task_config["GENVERSION"] = f"{prefix}_{sim_name}"
            sim_output_dir = f"{base_output_dir}/{stage_number}_SIM/{sim_name}"
            s = SNANASimulation(sim_name, sim_output_dir, task_config, global_config)
            Task.logger.debug(
                f"Creating simulation task {sim_name} with {s.num_jobs} jobs, output to {sim_output_dir}"
            )
            tasks.append(s)
        return tasks
