#!/bin/bash

#SBATCH --job-name=train_earthquake
#SBATCH --gpus=1
#SBATCH --time=24:00:00



module load cudatoolkit/23.9_12.2
conda activate SMASH

cd scripts

echo Working Directory: $(pwd)
echo Start Time: $(date)

bash train_${1}.sh "$2" "$3"

echo End Time: $(date)
