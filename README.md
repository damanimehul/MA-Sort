# Multi-Agent Sort (MA-Sort) 
This is the official codebase of MA-Sort, a MARL enviornment for comparison-based sorting. This work aims to replicate the emergence of dominance hierarchies in animal social groups when they compete for resources, resulting in the creation of a ranking system. 

## Setting up the codebase
Install conda environment using - 
```
conda env create -f requirements.yaml 
```
 
## Running the code
All parameters and their default values can be found in **main.py**.
This codebase supports A2C and multiple observation types in the MA-Sort environment. 
A simple job can be run as - 
```
python main.py --exp_name simple_run --save_gifs --n 4 --multi_policy 
```
The codebase has support for wandb and also logs to a .csv file, gifs can be saved regularly for visualization if required. If new parameters need to be added, they should be added to **main.py**, **setters.py** and the desired low-level file.

If multiple jobs need to be created, **job_maker.py** can efficiently generate a **config.json** file that contains multiple configurations to run. It also generates slurm scripts that make it easy to run batch jobs. If running such batch jobs, **full_runner.py** wraps around **main.py** and can efficiently read and run from the generated config.json. Sample scripts can be found inside **run_scripts/**. 

## Visualizations
Gifs are provided in **visualizations/** from some of the trainings run so far. Note that the rendering code has been set up such that the order of strengths is 1) maroon (strongest agent) 2) red 3) pink 4) blue. 