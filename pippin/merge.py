import shutil
import subprocess

from pippin.aggregator import Aggregator
from pippin.config import chown_dir, mkdirs
from pippin.snana_fit import SNANALightCurveFit
from pippin.task import Task
import os


class Merger(Task):
    def __init__(self, name, output_dir, dependencies, options):
        super().__init__(name, output_dir, dependencies=dependencies)
        self.options = options
        self.passed = False
        self.logfile = os.path.join(self.output_dir, "output.log")
        self.cmd_prefix = ["combine_fitres.exe", "t"]
        self.cmd_suffix = ["-outprefix", "merged"]
        self.done_file = os.path.join(self.output_dir, "merged.text")
        self.lc_fit = self.get_lcfit_dep()
        self.agg = self.get_agg_dep()

    def get_lcfit_dep(self):
        for d in self.dependencies:
            if isinstance(d, SNANALightCurveFit):
                return d.output
        msg = f"No dependency of a light curve fit task in {self.dependencies}"
        self.logger.error(msg)
        raise ValueError(msg)

    def get_agg_dep(self):
        for d in self.dependencies:
            if isinstance(d, Aggregator):
                return d.output
        msg = f"No dependency of an aggregator task in {self.dependencies}"
        self.logger.error(msg)
        raise ValueError(msg)

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(f"Merger finished, see combined fitres at {self.done_file}")

            # Copy MERGE.LOG and FITOPT.README if they aren't there
            filenames = ["MERGE.LOG", "FITOPT.README"]
            for f in filenames:
                original = os.path.join(self.lc_fit["lc_output_dir"], f)
                moved = os.path.join(self.output_dir, f)
                if not os.path.exists(moved):
                    self.logger.debug(f"Copying file {f} into output directory")
                    shutil.copy(original, moved)

            # Dick around with folders and names to make it resemble split_and_fit output for salt2mu
            outdir = os.path.join(self.output_dir, self.lc_fit["genversion"])
            new_output = os.path.join(outdir, "FITOPT000.FITRES")
            if not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)

                original_output = self.done_file
                shutil.move(original_output, new_output)

                # Create symlinks for all systematics
                original_dir = self.lc_fit["fitres_dir"]
                sys_files = [a for a in os.listdir(original_dir) if "FITOPT000" not in a and ".FITRES" in a]
                for s in sys_files:
                    os.symlink(os.path.join(original_dir, s), os.path.join(outdir, s))

                # Recreate done file -_-
                with open(self.done_file, "w") as f:
                    f.write("SUCCESS")

            self.output["fitres_file"] = new_output
            self.output["fitres_dir"] = outdir
            return Task.FINISHED_SUCCESS
        else:
            output_error = False
            if os.path.exists(self.logfile):
                with open(self.logfile, "r") as f:
                    for line in f.read().splitlines():
                        if "ERROR" in line or "ABORT" in line:
                            self.logger.error(f"Fatal error in combine_fitres. See {self.logfile} for details.")
                            output_error = True
                        if output_error:
                            self.logger.info(f"Excerpt: {line}")
                if output_error:
                    self.logger.debug("Removing hash on failure")
                    os.remove(self.hash_file)
                    chown_dir(self.output_dir)
                    return Task.FINISHED_FAILURE
            else:
                self.logger.error("Combine task failed with no output log. Please debug")
                return Task.FINISHED_FAILURE

    def _run(self, force_refresh):
        command = self.cmd_prefix + [self.lc_fit["fitres_file"], self.agg["merge_key_filename"]] + self.cmd_suffix

        old_hash = self.get_old_hash()
        new_hash = self.get_hash_from_string(" ".join(command))

        if force_refresh or new_hash != old_hash:
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            self.logger.debug("Regenerating, running combine_fitres")
            self.save_new_hash(new_hash)
            with open(self.logfile, "w") as f:
                subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)
        else:
            self.logger.debug("Not regnerating")
        return True

