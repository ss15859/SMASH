#!/bin/bash

# Define the parameter values
datasets=("ComCat" "WHITE")
seq_lens=(500 1000 2000)
marked_output=(0 1)

# Loop over all combinations and submit jobs
for dataset in "${datasets[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        for flag in "${marked_output[@]}"; do
            sbatch --output="slurm_outputs/${dataset}_seq_len_${seq_len}_marked_output_${flag}.out" job.sh "$dataset" "$seq_len" "$flag"
        done
    done
done
