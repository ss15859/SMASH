import os
import subprocess
from datetime import datetime, timedelta

# Define the parameters
dataset = "ComCat"  # Change this to the desired dataset
test_nll_start = datetime.strptime("2007-01-01", "%Y-%m-%d")
test_nll_end = datetime.strptime("2020-01-17", "%Y-%m-%d")
batch_size = 768  # for BC4

# Calculate the number of days between start and end dates
num_days = (test_nll_end - test_nll_start).days

# Loop over each day and submit a job
for day_number in range(num_days):
    command = ["sbatch", "job.sh", dataset, str(day_number), str(batch_size)]
    # subprocess.run(command)
    print(command)
    print(f"Submitted job for day {day_number}")
