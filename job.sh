#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_SMASH
#SBATCH --partition=gpu
#SBATCH --account=MATH026082

source /user/work/ss15859/miniforge3/etc/profile.d/conda.sh
conda activate earthquakeNPP

cd scripts

echo Working Directory: $(pwd)
echo Start Time: $(date)

bash train_${1}.sh "$2" "$3"

echo End Time: $(date)