#!/bin/bash
#SBATCH --job-name=fouragents-f-nr-s-v1
#SBATCH --open-mode=append
#SBATCH --output=/home/gridsan/mdamani/6.7950-project/slurm_jobs/fouragents-f-nr-s-v1/%x_%j.out
#SBATCH --error=/home/gridsan/mdamani/6.7950-project/slurm_jobs/fouragents-f-nr-s-v1/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH -c 1
#SBATCH -N 1
module purge
source ~/.bashrc
module load anaconda/2021b
cd $HOME/6.7950-project
python full_runner.py --path run_scripts/fouragents-f-nr-s-v1_2022-12-02
