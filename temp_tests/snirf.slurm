#!/bin/bash
#SBATCH --job-name=PA_SNIRF__SNIRFTEST_D_DATADES3YR_SNIRFTRAIN_D_DESTRAIN
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=broadwl
#SBATCH --output=output.log
#SBATCH --account=pi-rkessler
#SBATCH --mem=14GB

source activate snn_gpu
echo `which python`
cd /project2/rkessler/PRODUCTS/classifiers/snirf
python SNIRF.py --nc 4 --nclass 2 --ft x1 c zHD x1ERR cERR PKMJDERR --restore --pklfile /scratch/midway2/rkessler/PIPPIN_OUTPUT/GLOBAL/3_CLAS/SNIRFTRAIN_D_DESTRAIN/model.pkl --pklformat FITRES --test /scratch/midway2/rkessler/PIPPIN_OUTPUT/PA_SNIRF/2_LCFIT/D_DATADES3YR/output/DESALL_specType_SMP_real_snana_fits/FITOPT000.FITRES.gz --filedir /scratch/midway2/rkessler/PIPPIN_OUTPUT/PA_SNIRF/3_CLAS/SNIRFTEST_D_DATADES3YR_SNIRFTRAIN_D_DESTRAIN --done_file /scratch/midway2/rkessler/PIPPIN_OUTPUT/PA_SNIRF/3_CLAS/SNIRFTEST_D_DATADES3YR_SNIRFTRAIN_D_DESTRAIN/done.txt --use_filenames 
