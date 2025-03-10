import os
from datetime import datetime, timedelta
import argparse

# Define the parameters for each dataset
datasets = {
    "ComCat": {
        "test_nll_start": "2007-01-01",
        "test_nll_end": "2020-01-17"
    },
    "WHITE": {
        "test_nll_start": "2017-01-01",
        "test_nll_end": "2021-01-01"
    },
    "SCEDC": {
        "test_nll_start": "2014-01-01",
        "test_nll_end": "2020-01-01"
    },
    "SanJac": {
        "test_nll_start": "2016-01-01",
        "test_nll_end": "2018-01-01"
    },
    "SaltonSea": {
        "test_nll_start": "2016-01-01",
        "test_nll_end": "2018-01-01"
    }
}

# Parse command line arguments
parser = argparse.ArgumentParser(description="Submit jobs for each test day between test_nll_start and test_nll_end.")
parser.add_argument("--dataset", type=str, required=True, choices=datasets.keys(), help="Dataset name")
parser.add_argument("--batch_size", type=int, default=768, help="Batch size for the jobs")
args = parser.parse_args()

# Get the parameters for the selected dataset
dataset_params = datasets[args.dataset]
test_nll_start = datetime.strptime(dataset_params["test_nll_start"], "%Y-%m-%d")
test_nll_end = datetime.strptime(dataset_params["test_nll_end"], "%Y-%m-%d")
batch_size = args.batch_size

# Calculate the number of days between start and end dates
num_days = (test_nll_end - test_nll_start).days

# Loop over each day and submit a job
for day_number in range(num_days):
    if args.dataset == "ComCat":
        # Skip the first 436 days for ComCat
        if day_number <= 436:
            continue
    command = f"sbatch --output=slurm_outputs/{args.dataset}_day_{day_number}.out job.sh {args.dataset} {day_number} {batch_size}"
    os.system(command)
    print(f"Submitted job for day {day_number}")
    