#!/bin/bash
#SBATCH --time=35:30:00
#SBATCH --partition=broadwl
#SBATCH --account=pi-rkessler  
#SBATCH --job-name=COSMOMC_DJB
#SBATCH --output=sn_%j.error
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 126000 = 35 hours

