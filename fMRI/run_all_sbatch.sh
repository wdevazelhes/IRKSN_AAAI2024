#!/bin/bash
#SBATCH --job-name=your_job_name    # Job name
#SBATCH --output=slurm-%N-%j.out   # Standard output and error log
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=32        # Number of CPU cores per task
#SBATCH -q cpu-512                 # Queue name
#SBATCH --mem=40G                  # Total RAM to be used
#SBATCH --time=12:00:00            # Specify the time needed for your experiment

if [ "$#" -ne 3 ]; then
  echo "Usage: sbatch your_sbatch_script.sh <class1> <class2> <num_threads>"
  exit 1
fi

# Set class1, class2, and num_threads from command-line arguments
class1="$1"
class2="$2"
num_threads="$3"

python global_script.py lasso "$class1" "$class2" "$num_threads" &
python global_script.py enet "$class1" "$class2" "$num_threads" &
python global_script.py omp "$class1" "$class2" "$num_threads" &
python global_script.py iht "$class1" "$class2" "$num_threads" &
python global_script.py ksn "$class1" "$class2" "$num_threads" &
python global_script.py irksn "$class1" "$class2" "$num_threads" &
python global_script.py ircr "$class1" "$class2" "$num_threads" &
python global_script.py irosr "$class1" "$class2" "$num_threads" &
python global_script.py srdi "$class1" "$class2" "$num_threads" &
python encludl.py "$class1" "$class2" "$num_threads" &
wait