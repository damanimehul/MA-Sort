import numpy as np
import os.path as osp, time, atexit, os
import wandb 

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class Logger:
    """
    A general-purpose logger.
    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self,args,output_fname='progress.csv'):
        """
        Initialize a Logger.
        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.
            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 
            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        self.output_dir = args.output_dir 
        if osp.exists(self.output_dir):
            print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
            self.gif_directory = self.output_dir +'/gifs/'
        else:
            os.makedirs(self.output_dir)
            self.gif_directory = self.output_dir +'/gifs/'
            os.makedirs(self.gif_directory) 
        self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
        atexit.register(self.output_file.close)
        print(colorize("Logging data to %s"%self.output_file.name, 'green', bold=True))
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = args.exp_name
        self.wandb_logger=Wandb_Logger(args) 

    def log(self,vals_dict,step):
        num_elements = len(vals_dict)
        count,commit = 0 , False 
        for key,val in vals_dict.items() : 
            count+=1 
            if count == num_elements :
                commit = True  
            self.log_tabular(key,val,step,commit )
        
    def log_tabular(self, key, val,step=None,commit=False):
        """
        Log a value of some diagnostic.
        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val
        self.wandb_logger.log(key,val,step,commit)

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.
        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15,max(key_lens))
        keystr = '%'+'%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-"*n_slashes)
        for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g"%val if hasattr(val, "__float__") else val
                print(fmt%(key, valstr))
                vals.append(val)
        print("-"*n_slashes, flush=True)
        if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers)+"\n")
                self.output_file.write("\t".join(map(str,vals))+"\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False

class Wandb_Logger() :
    def __init__(self,args) :
        self.update = True  
        self.wandb_run = None 
        if args.wandb: 
            name = args.exp_name + time.strftime("_%Y-%m-%d") 
            self.wandb_run = wandb.init(project='6.7950',config=args,tags=[args.exp_name],name=name,settings=wandb.Settings(start_method="fork",_disable_stats=True))  
        if self.wandb_run is None :
            self.update = False 
        self.logs = { } 

    def log(self,key=None,val=None,step=None,commit=False) :
        if self.update :
            self.log_single(key,val,step,commit) 

    def log_single(self,key,val,step,commit) : 
        assert key is not None 
        if not commit : 
            self.logs[key] = val 
        else :
            assert step is not None 
            self.logs[key] = val  
            self.wandb_run.log(self.logs ,step=step) 
            self.logs = {} 