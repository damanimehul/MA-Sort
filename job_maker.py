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
    parser.add_argument('--seeds',type=int,default=1,help='Number of seeds for each run')

    args = parser.parse_args() 
    return args 

def get_configs(grid) :
    final_configs =[] 
    for i in range(len(grid)) :
        desired_config = '' 
        config = grid[i] 
        for k,v in config.items() :
            if v !=None :
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
    grid = {"multi_policy":['',None],"n":[2,4],"gae_lambda":[0,0.5],"shuffle_ranks":['',None],"memory":[''],"fights_info":['']} 
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

    config_dict,details = {} , {'base':args.params} 
    
    for i,setup in enumerate(configs) :   
        details[i] = setup 
        for seed in range(args.seeds) : 
            experiment_name = args.name + '-' + str(len(config_dict)) + '-s' + str(seed) 
            config_dict[len(config_dict)] = str(' --exp_name {} --seed {} --tag {} '.format(experiment_name,seed,args.name) +  '{} {} '.format(args.params,setup))
                          
encoded_configs = jsonpickle.encode(config_dict) 
path = "run_scripts/" + args.name + "_" + time.strftime("%Y-%m-%d") 
os.makedirs(path) 

encoded_details = jsonpickle.encode(details)  

with open(path+"/config.json","w") as file :
    json.dump(encoded_configs,file) 
    file.close() 

with open(path+"/details.json","w") as file :
    json.dump(encoded_details,file) 
    file.close() 

with open(path+"/runner.sh","w") as file : 
    file.write('mkdir /home/gridsan/mdamani/6.7950-project/slurm_jobs/{}\n'.format(args.name)) 
    file.write('sbatch job.slurm ') 
    file.close() 

with open(path+"/job.slurm","w") as f :
    f.write('#!/bin/bash\n') 
    f.write('#SBATCH --job-name={}\n'.format(experiment_name)) 
    f.write('#SBATCH --open-mode=append\n') 
    f.write('#SBATCH --output=/home/gridsan/mdamani/6.7950-project/slurm_jobs/{}/%a_%x_%j.out\n'.format(args.name)) 
    f.write('#SBATCH --error=/home/gridsan/mdamani/6.7950-project/slurm_jobs/{}/%a_%x_%j.err\n'.format(args.name)) 
    f.write('#SBATCH --export=ALL\n') 
    f.write('#SBATCH --time=30:00:00\n')
    f.write('#SBATCH --mem=8G\n')
    f.write('#SBATCH -c 1\n') 
    f.write('#SBATCH --array=1-{}\n'.format(len(config_dict)) ) 
    f.write('module purge\n')
    f.write('source ~/.bashrc\n') 
    f.write('module load anaconda/2021b\n') 
    f.write('cd $HOME/6.7950-project\n')
    f.write('python full_runner.py --path {} --array_id $SLURM_ARRAY_TASK_ID\n'.format(path)) 