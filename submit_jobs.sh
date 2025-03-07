#!/bin/bash

# Define the parameter values
datasets=("SCEDC" "SaltonSea" "SanJac")

# Loop over all combinations and submit jobs
for dataset in "${datasets[@]}"; do
    sbatch --output="slurm_outputs/${dataset}.out" job.sh "$dataset"
done
