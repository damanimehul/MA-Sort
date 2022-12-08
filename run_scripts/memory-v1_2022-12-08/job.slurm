#!/bin/bash
#SBATCH --job-name=memory-v1-47-s2
#SBATCH --open-mode=append
#SBATCH --output=/home/gridsan/mdamani/6.7950-project/slurm_jobs/memory-v1/%a_%x_%j.out
#SBATCH --error=/home/gridsan/mdamani/6.7950-project/slurm_jobs/memory-v1/%a_%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=30:00:00
#SBATCH --mem=8G
#SBATCH -c 1
#SBATCH --array=1-48
module purge
source ~/.bashrc
module load anaconda/2021b
cd $HOME/6.7950-project
python full_runner.py --path run_scripts/memory-v1_2022-12-08 --array_id $SLURM_ARRAY_TASK_ID
