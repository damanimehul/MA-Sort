import json 
import jsonpickle 
import os 
if __name__ =="__main__" :
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--path', type=str , default = None , help = 'Specify the path to benchmark config file') 
    parser.add_argument('--array_id', type = int, default = 1, help = 'Enables running of a batch job')
    args = parser.parse_args() 
    assert args.path is not None 
    assert args.array_id is not None 
    with open(args.path+"/config.json","r") as jsonfile:
        configurations = dict(jsonpickle.decode(json.load(jsonfile))) 
        jsonfile.close() 

    config = configurations[str(args.array_id-1)]  
    print(config)
    os.system('python main.py ' + config)


