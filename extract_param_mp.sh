#!/bin/bash
#SBATCH --job-name=extract_param
#SBATCH --partition=gpu  # submit to the serial queue
#SBATCH --time=2-00:00:00  # Maximum wall time for the job
#SBATCH --account=Research_Project-T127204  # research project to submit under
#SBATCH --nodes=1  # specify number of nodes
#SBATCH --ntasks-per-node=1  # specify the number of tasks per node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24  # cpus per task
#SBATCH --mem-per-cpu=8G  # GB memory requested per cpu-core
#SBATCH --output=extract_param.out  # submit script's standard-out
#SBATCH --error=extract_param.err  # submit script's standard-error

# Load necessary modules and activate conda environment
source ~/.bashrc
conda activate interactive_head

cd /lustre/projects/Research_Project-T127204/xk219/projects/ai_digital_humans_repo_summary/smirk

srun python extract_param_mp.py