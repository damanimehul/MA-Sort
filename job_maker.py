import sys
import argparse 
import time 
import jsonpickle 
import json 
import os 
import itertools

def get_args() :
    parser= argparse.ArgumentParser() 
    parser.add_argument('--name' ,type = str , default=None , help= 'Name for campaign to run') 
    parser.add_argument('--params', type = str, default= None , help='Other specific params in a string eg. --others "--neptune --hello"')  
    parser.add_argument('--use_grid' , action='store_true', default = False , help = 'Use itertools to get all combinations of items defined in grid')
    args = parser.parse_args() 
    return args 

def get_configs(grid) :
    final_configs =[] 
    for i in range(len(grid)) :
        desired_config = '' 
        config = grid[i] 
        for k,v in config.items() :
            desired_config += ' --' + str(k) + ' ' + str(v)  

        final_configs.append(desired_config)
    return final_configs

def remove_ignore_configs(grid,ignore) :
    # NOT USED FOR NOW 
    indexes_to_remove = [] 
    for i in range(len(ignore)): 
        ignore_config = ignore[i] 
        for j in range(len(grid)) : 
            remove= True 
            config = grid[j] 
            for k,v in ignore_config.items() :
                if type(v) == list :
                    if config[k] not in v :
                        remove = False 
                        break 
                elif v != config[k] : 
                    remove= False 
                    break
            if remove ==True :
                indexes_to_remove.append(j) 
    new_grid =[] 
    for index in range(len(grid)) :
        if index not in indexes_to_remove :
            new_grid.append(grid[index]) 
    return new_grid 
        
if __name__ == '__main__': 
    grid = {"v_fit_rate":[150],"current_ability":['target'],"target":['old_target'],"resample_frequency":[20],"her_ratio":[3]} 
    configs = [''] 
    args = get_args() 

    if args.use_grid is False : 
        grid =[''] 
    else :
        ignore_configs =[] 
        grid_setups = list(dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values()))
        grid_setups = remove_ignore_configs(grid_setups,ignore_configs)
        configs = get_configs(grid_setups)
 
    experiment_name = args.name  
    experiment_arguments = ''
    if args.params is not None:
        experiment_arguments = args.params 

    config_dict = {} 
 
    for setup in configs :  

        config_dict[len(config_dict)] = str(' --exp_name {}'.format(experiment_name) +  '{}'.format(args.params))
                          
encoded_configs = jsonpickle.encode(config_dict) 
path = "run_scripts/" + experiment_name + "_" + time.strftime("%Y-%m-%d") 
os.makedirs(path) 
encoded_details = jsonpickle.encode({'grid':grid , 'use_grid':args.use_grid}) 

with open(path+"/config.json","w") as file :
    json.dump(encoded_configs,file) 
    file.close() 

with open(path+"/runner.sh","w") as file : 
    file.write('mkdir /home/gridsan/mdamani/6.7950-project/slurm_jobs/{}\n'.format(args.name)) 
    file.write('sbatch job.slurm ') 
    file.close() 

with open(path+"/job.slurm","w") as f :
    f.write('#!/bin/bash\n') 
    f.write('#SBATCH --job-name={}\n'.format(experiment_name)) 
    f.write('#SBATCH --open-mode=append\n') 
    f.write('#SBATCH --output=/home/gridsan/mdamani/6.7950-project/slurm_jobs/{}/%x_%j.out\n'.format(args.name)) 
    f.write('#SBATCH --error=/home/gridsan/mdamani/6.7950-project/slurm_jobs/{}/%x_%j.err\n'.format(args.name)) 
    f.write('#SBATCH --export=ALL\n') 
    f.write('#SBATCH --time=12:00:00\n')
    f.write('#SBATCH --mem=8G\n')
    f.write('#SBATCH -c 1\n') 
    f.write('#SBATCH -N 1\n') 
    f.write('module purge\n')
    f.write('source ~/.bashrc\n') 
    f.write('module load anaconda/2021b\n') 
    f.write('cd $HOME/6.7950-project\n')
    f.write('python full_runner.py --path {}\n'.format(path)) 