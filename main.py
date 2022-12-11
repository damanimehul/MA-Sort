import argparse 
import torch 
import setter
import numpy as np 

if __name__ =='__main__' : 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--exp_name',type=str,default='Simple-Run',help='Give a name to the experiment')
    parser.add_argument('--tag',type=str,default=None,help='if doing a batch job, then this should be the name of the batch job. For a single run it is just the exp_name')
    parser.add_argument('--obs_type',type=str,default='features',help='Select the observation type between grid/features/both')
    parser.add_argument('--n',type=int,default=4,help='Number of agents to make the environment from')
    parser.add_argument('--seed',type=int,default=0,help='Number of agents to make the environment from')
    parser.add_argument('--random_init',action='store_true',default=False,help='Agent positions are sampled randomly on the grid') 
    parser.add_argument('--save_gifs',action='store_true',default=False,help='GIFS are saved periodically every gif_frequency steps') 
    parser.add_argument('--gif_frequency',type=int,default=500,help='New gif every 50 episodes')
    parser.add_argument('--gamma',type=float,default=0.995,help='Discount Factor')
    parser.add_argument('--train_episodes',type=int,default=20000,help='Number of episodes to train for')
    parser.add_argument('--wandb',action='store_true',default=False,help='Log on wandb')
    parser.add_argument('--single_agent',action='store_true',default=False,help='Single centralized agent or not')
    parser.add_argument('--max_ep_len',type=int,default=50,help='Maximum episode length')
    parser.add_argument('--lr',type=float,default=5e-5,help='Learning rate') 
    parser.add_argument('--hidden_sizes',type=list,default=[128,128],help='Hidden dizes for the shared actor critic network')
    parser.add_argument('--fights_info',action='store_true',default=False,help='Whether or not to provide information about total fights and fights won if using feature based obs ')
    parser.add_argument('--shuffle_ranks',action='store_true',default=False,help='If no shuffling, then the id of the agent already provides information about the optimal index, using this as a baseline to validate the learning algo ')
    parser.add_argument('--v_coeff',type=float,default=0.01,help='Coefficient for value loss')
    parser.add_argument('--log_freq',type=int,default=100,help='Frequency of logging, can basically be thought of as an epoch length')
    parser.add_argument('--norm_rewards',action='store_true',default=False,help='Normalize rewards ')
    parser.add_argument('--multi_policy',action='store_true',default=False,help='Share policy parameters or not, if sharing then the policy returned by setter would be a dictionary containing policy objects ')
    parser.add_argument('--train_freq',type=int,default=10,help='How many episodes to rollout before calling algo.train() ')
    parser.add_argument('--algo',type=str,default='a2c',help='Which algorithm to run between A2C/PPO ')
    parser.add_argument('--memory',action='store_true',default=False,help='Give agents a feature which provides the agents best current rank (basically providing the soln)')
    parser.add_argument('--ent_coeff',type=float,default=0,help='Coefficient on entropy loss')
    args = parser.parse_args()  

    if args.tag is None : 
        args.tag = args.exp_name 
    args.output_dir = 'results/{}/{}'.format(args.tag,args.exp_name)

    # Setting required seeds for random number generators 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.seed !=0 : 
        args.save_gifs = False 

    env = setter.set_env(args) 
    policy = setter.set_policy(args,env) 
    buffer = setter.set_buffer(args,env) 
    logger = setter.set_logger(args) 
    algo = setter.set_algo(args,policy,buffer,env,logger) 

    ep = 0 
    while ep < args.train_episodes : 
        ep_stats = algo.collect_rollout()  
        logger.store(ep_stats) 
        ep+=1 
        if ep%args.train_freq ==0 : 
            train_stats = algo.train() 
            logger.store(train_stats) 

        if ep%args.log_freq ==0 : 
            logger.log_tabular(ep_stats,step=ep) 
            logger.log_tabular(train_stats,step=ep) 
            if args.seed ==0 :
                img_stats = algo.get_img_stats() 
                logger.image_log(img_stats,step=ep)
            logger.dump_tabular() 

    