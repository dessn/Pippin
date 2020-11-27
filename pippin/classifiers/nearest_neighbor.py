import subprocess
import os
import shutil
import tempfile
import glob
from pippin.classifiers.classifier import Classifier
from pippin.config import mkdirs, get_output_loc, copytree
from pippin.task import Task


class NearestNeighborClassifier(Classifier):
    """ Nearest Neighbour classifier

    CONFIGURATION
    =============

    CLASSIFICATION:
        label:
            MASK_SIM: mask  # partial match
            MASK_FIT: mask  # partial match
            MASK: mask  # partial match
            MODE: train/predict
            OPTS:  # Options
                FITOPT: fitoptLabel  # Exact match
                MODEL: someName # exact name of training classification task

    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        prob_column_name: name of the column to get probabilities out of
        predictions_filename: location of csv filename with id/probs

    """

    def __init__(self, name, output_dir, config, dependencies, mode, options, index=0, model_name=None):
        super().__init__(name, output_dir, config, dependencies, mode, options, index=index, model_name=model_name)
        self.passed = False
        self.num_jobs = 40
        self.outfile_train = f"{output_dir}/NN_trainResult.out"
        self.outfile_predict = f"{output_dir}/predictions.out"
        self.logging_file = os.path.join(output_dir, "output.log")
        self.splitfit_output_dir = f"{self.output_dir}/output"
        self.options = options

        self.num_check = 0
        self.max_check = 5
        self.options = options
        self.nn_options = "z .01 .12 .01 c 0.01 0.19 .01 x1 0.1 1.1 .04"
        self.train_info_local = {}

    def setup(self):
        lcfit = self.get_fit_dependency()
        self.fitopt = self.options.get("FITOPT", "DEFAULT")
        self.fitres_filename = lcfit["fitopt_map"][self.fitopt]
        self.fitres_path = os.path.abspath(os.path.join(lcfit["fitres_dirs"][self.index], self.fitres_filename))

    def train(self):
        # Created April 2019 by R.Kessler
        # Train nearest nbr.
        self.setup()
        # prepare new split-and_fit NML file with extra NNINP namelist
        new_hash, self.train_info_local = self.prepare_train_job()
        self.output["model_filename"] = self.outfile_train
        if new_hash is None:
            return True
        if self.train_info_local is None:
            return False

        # run split_and_fit job
        self.run_train_job()
        return True

    def prepare_train_job(self):
        self.logger.debug("Preparing NML file for Nearest Neighbour training")
        fit_output = self.get_fit_dependency()
        self.num_jobs = self.get_fit_dependency(output=False).num_jobs

        genversion = fit_output["genversion"]
        fitres_dir = fit_output["fitres_dirs"][self.index]
        fitres_file = self.fitres_path
        nml_file_orig = fit_output["nml_file"]

        # Put config in a temp directory
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name

        outfile_train = f"{self.name}_train.out"
        nml_file_train1 = f"{temp_dir}/{genversion}-2.nml"
        nml_file_train2 = f"{self.output_dir}/{genversion}-2.nml"

        train_info_local = {"outfile_NNtrain": outfile_train, "nml_file_NNtrain": nml_file_train2}

        # construct sed to copy original NMLFILE and to
        #   + replace OUTDIR:
        #   + include ROOTFILE_OUT (to store histograms for NN train)
        #   + include DONE stamp for Sam/pippen
        #   + run afterburner to process ROOT file and get NN_trainPar;
        #     copy NN_trainPar up to where pippin can find it
        #
        # TODO: Check with Rick if the FITOPT000.ROOT is needed / should be hardcoded
        afterBurn = f"nearnbr_maxFoM.exe FITOPT000.ROOT -truetype 1 -outfile {outfile_train} ; cp {outfile_train} {self.outfile_train}"

        sedstr = "sed"
        sedstr += r" -e '/OUTDIR:/a\OUTDIR: %s' " % self.splitfit_output_dir
        sedstr += r" -e '/OUTDIR:/d'"
        sedstr += r" -e '/DONE_STAMP:/d'"
        sedstr += r" -e '/SNTABLE_LIST/a\    ROOTFILE_OUT = \"bla.root\"'"
        sedstr += r" -e '/_OUT/d '"
        sedstr += r" -e '/VERSION:/a\VERSION_AFTERBURNER: %s'" % afterBurn
        sedstr += r" -e '/VERSION:/a\DONE_STAMP: %s'" % self.done_file
        sed_command = "%s %s > %s" % (sedstr, nml_file_orig, nml_file_train1)

        # use system call to apply sed command
        # self.logger.debug(f"Running sed command {sed_command}")
        subprocess.run(sed_command, stderr=subprocess.STDOUT, cwd=temp_dir, shell=True)

        # make sure that the new NML file is really there
        if not os.path.isfile(nml_file_train1):
            self.logger.error(f"Unable to create {nml_file_train1} with sed command {sed_command}")
            return None

        # check that expected FITRES ref file is really there.
        if not os.path.exists(fitres_file):
            self.logger.error("Cannot find expected FITRES file at {fitres_path}")
            return None

        # open NML file in append mode and tack on NNINP namelist
        with open(nml_file_train1, "a") as f:
            f.write("\n# NNINP below added by prepare_NNtrainJob\n")
            f.write("\n&NNINP \n")
            f.write("   NEARNBR_TRAINFILE_PATH = '%s' \n" % fitres_dir)
            f.write("   NEARNBR_TRAINFILE_LIST = '%s' \n" % os.path.basename(fitres_file))
            f.write("   NEARNBR_SEPMAX_VARDEF  = '%s' \n" % self.nn_options)
            f.write("   NEARNBR_TRUETYPE_VARNAME = 'SIM_TYPE_INDEX' \n")
            f.write("   NEARNBR_TRAIN_ODDEVEN = T \n")
            f.write("\n&END\n")

        input_files = [nml_file_train1]
        new_hash = self.get_hash_from_files(input_files)

        if self._check_regenerate(new_hash):
            self.logger.debug("Regenerating")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            self.logger.debug(f"Copying from {temp_dir} to {self.output_dir}")
            copytree(temp_dir, self.output_dir)
            self.save_new_hash(new_hash)
            return new_hash, train_info_local
        else:
            self.logger.info("Hash check passed, not rerunning")
            self.should_be_done()
            return None, train_info_local

    def run_train_job(self):
        cmd = ["split_and_fit.pl", self.train_info_local["nml_file_NNtrain"], "NOPROMPT"]
        self.logger.debug(f"Launching training via {cmd}")
        with open(self.logging_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)

    def _check_completion(self, squeue):
        outdir = self.splitfit_output_dir

        # check global DONE stamp to see if all is DONE
        if os.path.exists(self.done_file):
            self.logger.debug(f"Done file found at {self.done_file}")
            with open(self.done_file) as f:
                if "FAILURE" in f.read().upper():
                    self.logger.error("Done file has FAILURE stamp!")
                    return Task.FINISHED_FAILURE
            if self.mode == Classifier.TRAIN:
                tarball = os.path.join(outdir, "SPLIT_JOBS_LCFIT.tar.gz")
                if os.path.exists(tarball):
                    return Task.FINISHED_SUCCESS
                else:
                    # Because the tarball can be delayed, allow a few checks before failing
                    self.num_check += 1
                    if self.num_check >= self.max_check:
                        self.logger.error(f"Error, no tarball found at {tarball}")
                        return Task.FINISHED_FAILURE
                    else:
                        return 0
            else:
                if os.path.exists(self.outfile_predict):
                    self.logger.debug(f"Predictions can be found at {self.outfile_predict}")
                    self.output["predictions_filename"] = self.outfile_predict
                    return Task.FINISHED_SUCCESS
                else:
                    self.logger.error(f"No predictions found at {self.outfile_predict}")
                    return Task.FINISHED_FAILURE
        else:
            if os.path.exists(self.logging_file):
                with open(self.logging_file, "r") as f:
                    output_error = False
                    for line in f.read().splitlines():
                        if ("ERROR" in line or ("ABORT" in line and " 0 " not in line)) and not output_error:
                            self.logger.error(f"Fatal error in light curve fitting. See {self.logging_file} for details.")
                            output_error = True
                        if output_error:
                            self.logger.info(f"Excerpt: {line}")

                if output_error:
                    return Task.FINISHED_FAILURE

            if self.mode == Classifier.TRAIN:
                done_path = os.path.join(outdir, "SPLIT_JOBS_LCFIT")
                num_done = len(glob.glob1(done_path, "*.DONE"))
                num_remain = self.num_jobs - num_done
                return num_remain
            else:
                return 0

    def predict(self):
        self.setup()

        model = self.options.get("MODEL")
        assert model is not None, "If TRAIN is not specified, you have to point to a model to use"
        for t in self.dependencies:
            if model == t.name:
                self.logger.debug(f"Found task dependency {t.name} with model file {t.output['model_filename']}")
                model = t.output["model_filename"]

        model_path = get_output_loc(model)
        self.logger.debug(f"Looking for model in {model_path}")
        if not os.path.exists(model_path):
            self.logger.error(f"Cannot find {model_path}")
            return False

        new_hash = self.get_hash_from_string(self.name + model_path)
        if self._check_regenerate(new_hash):
            self.logger.debug("Regenerating")

            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            self.save_new_hash(new_hash)

            job_name = "nearnbr_apply.exe"
            inArgs = f"-inFile_data {self.fitres_path} -inFile_MLpar {model_path}"
            outArgs = f"-outFile {self.outfile_predict} -varName_prob {self.get_prob_column_name()}"
            cmd_job = "%s %s %s" % (job_name, inArgs, outArgs)
            self.logger.debug(f"Executing command {cmd_job}")
            with open(self.logging_file, "w") as f:
                val = subprocess.run(cmd_job.split(" "), stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)
                with open(self.done_file, "w") as f:
                    if val.returncode == 0:
                        f.write("SUCCESS")
                    else:
                        f.write("FAILURE")
        else:
            self.logger.debug("Not regenerating")
        return True
