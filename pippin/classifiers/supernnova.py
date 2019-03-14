import os
import inspect
import subprocess
from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs, get_config


class SuperNNovaClassifier(Classifier):
    def __init__(self, light_curve_dir, fit_dir, output_dir, options):
        super().__init__(light_curve_dir, fit_dir, output_dir, options)
        self.global_config = get_config()
        self.dump_dir = output_dir + "/dump"
        self.job_base_name = os.path.basename(output_dir)

        self.slurm = """
#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err
#SBATCH --account=pi-rkessler

source ~/.bashrc
conda activate {conda_env}
module load cuda
cd {path_to_supernnova}
python run.py --data --dump_dir {dump_dir} --raw_dir {photometry_dir} --fits_dir {fit_dir}
python run.py --use_cuda {command} --dump_dir {dump_dir}
        """
        self.conda_env = self.global_config["SuperNNova"]["conda_env"]
        self.path_to_supernnova = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/../../" + self.global_config["SuperNNova"]["location"])

    def classify(self):
        mkdirs(self.output_dir)
        if self.options.get("TRAIN"):
            return self.train()


        # fitres = f"{self.fit_dir}/FITOPT000.FITRES.gz"
        # self.logger.debug(f"Looking for {fitres}")
        # if not os.path.exists(fitres):
        #     self.logger.error(f"FITRES file could not be found at {fitres}, classifer has nothing to work with")
        #     return False
        #
        # data = pd.read_csv(fitres, sep='\s+', comment="#", compression="infer")
        # ids = data["CID"].values
        # probability = np.random.uniform(size=ids.size)
        # combined = np.vstack((ids, probability)).T
        #
        # output_file = self.output_dir + "/prob.txt"
        # self.logger.info(f"Saving probabilities to {output_file}")
        # np.savetxt(output_file, combined)
        # chown_dir(self.output_dir)
        return True # change to hash

    def train(self):
        format_dict = {
            "conda_env": self.conda_env,
            "dump_dir": self.dump_dir,
            "photometry_dir": self.light_curve_dir,
            "fit_dir": self.fit_dir,
            "path_to_supernnova": self.path_to_supernnova,
            "job_name": f"train_{self.job_base_name}",
            "command": "--train_rnn"
        }

        slurm_output_file = self.output_dir + "/train_job.slurm"
        self.logger.info(f"Training SuperNNova, slurm job outputting to {slurm_output_file}")

        with open(slurm_output_file, "w") as f:
            f.write(self.slurm.format(**format_dict))

        self.logger.info("Submitting batch job to train SuperNNova")
        subprocess.run(["sbatch", "--wait", slurm_output_file], cwd=self.output_dir)
        self.logger.info("Batch job finished")
        chown_dir(self.output_dir)
