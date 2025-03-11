#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --job-name=train_SMASH
#SBATCH --account=MATH026082

source /user/work/ss15859/miniforge3/etc/profile.d/conda.sh
conda activate earthquakeNPP

cd scripts

echo Working Directory: $(pwd)
echo Start Time: $(date)

bash train_${1}.sh "$2" "$3"

echo End Time: $(date)