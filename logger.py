import numpy as np
import os.path as osp, time, atexit, os
import wandb
import os 
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

def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    # global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = np.min(x) if len(x) > 0 else np.inf
        # global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = np.max(x) if len(x) > 0 else -np.inf
        # global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std

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
        self.output_dir = args.output_dir + time.strftime("_%Y-%m-%d") 
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

class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.
    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to 
    report the average / std / min / max value of that quantity.
    With an EpochLogger, each time the quantity is calculated, you would
    use 
    .. code-block:: python
        epoch_logger.store(NameOfQuantity=quantity_value)
    to load it into the EpochLogger's state. Then at the end of the epoch, you 
    would use 
    .. code-block:: python
        epoch_logger.log_tabular(NameOfQuantity, **options)
    to record the desired values.
    """

    def __init__(self,args,output_fname='progress.csv'):
        super().__init__(args=args,output_fname=output_fname)
        self.epoch_dict = dict()

    def store(self, vals):
        """
        Save something into the epoch_logger's current state.
        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k,v in vals.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, vals_dict,step, with_min_and_max=False, average_only=True):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.
        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.
            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.
            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.
            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        num_elements = len(vals_dict)
        count,commit = 0 , False 
        for key,val in vals_dict.items() : 
            count+=1 
            if count == num_elements :
                commit = True  
            if val is not None:
                v = self.epoch_dict[key]
                vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
                stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
                super().log_tabular(key if average_only else 'Average' + key, stats[0],step,commit)
                if not(average_only):
                    super().log_tabular('Std'+key, stats[1],step,commit)
                if with_min_and_max:
                    super().log_tabular('Max'+key, stats[3],step,commit)
                    super().log_tabular('Min'+key, stats[2],step,commit)
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        return mpi_statistics_scalar(vals)

    def image_log(self,img_dict,step) : 
        for k,v in img_dict.items() : 
            self.wandb_logger.log(key=k,val=wandb.Image(v),step = step,commit=True) 

class Wandb_Logger() :
    def __init__(self,args) :
        self.update = True  
        self.wandb_run = None 
        if args.wandb: 
            name = args.exp_name + time.strftime("_%Y-%m-%d") 
            os.environ["WANDB_MODE"] = "offline" 
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