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
parser.add_argument("--CPU", type=bool, default=False, help="Run on CPU")
args = parser.parse_args()

# Get the parameters for the selected dataset
dataset_params = datasets[args.dataset]
test_nll_start = datetime.strptime(dataset_params["test_nll_start"], "%Y-%m-%d")
test_nll_end = datetime.strptime(dataset_params["test_nll_end"], "%Y-%m-%d")
batch_size = args.batch_size

# Calculate the number of days between start and end dates
num_days = (test_nll_end - test_nll_start).days

if args.CPU:
    cpu_string = "CPU"
else:
    cpu_string = ""

# Loop over each day and submit a job
for day_number in range(num_days):

    # check if forecast already exists
    forecast_file = f"/user/work/ss15859/SMASH_daily_forecasts/{args.dataset}/CSEP_day_{day_number}.csv"
    if os.path.exists(forecast_file):
        print(f"Forecast for day {day_number} already exists")
        continue

    command = f"sbatch --output=slurm_outputs/{args.dataset}_day_{day_number}.out {cpu_string}job.sh {args.dataset} {day_number} {batch_size}"
    os.system(command)