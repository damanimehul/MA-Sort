import argparse 
import torch 
import setter

if __name__ =='__main__' : 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--exp_name',type=str,default='Simple-Run',help='Give a name to the experiment')
    parser.add_argument('--obs_type',type=str,default='features',help='Select the observation type between grid/features/both')
    parser.add_argument('--n',type=int,default=4,help='Number of agents to make the environment from')
    parser.add_argument('--random_init',type=int,default=1,help='Agent positions are sampled randomly on the grid') 
    parser.add_argument('--save_gifs',type=int,default=False,help='GIFS are saved periodically every gif_frequency steps') 
    parser.add_argument('--gif_frequency',type=int,default=50,help='New gif every 50 episodes')
    parser.add_argument('--gamma',type=float,default=0.995,help='Discount Factor')
    parser.add_argument('--train_episodes',type=int,default=10000,help='Number of episodes to train for')
    parser.add_argument('--wandb',action='store_true',default=False,help='Log on wandb')
    parser.add_argument('--single_agent',action='store_true',default=False,help='Log on wandb')
    parser.add_argument('--max_ep_len',type=int,default=100,help='Maximum episode length')
    parser.add_argument('--lr',type=float,default=5e-4,help='Learning rate') 
    parser.add_argument('--hidden_sizes',type=list,default=[128,128],help='Hidden dizes for the shared actor critic network')
    parser.add_argument('--fights_info',type=int,default=1,help='Whether or not to provide information about total fights and fights won if using feature based obs ')
    parser.add_argument('--shuffle_ranks',type=int,default=1,help='If no shuffling, then the id of the agent already provides information about the optimal index, using this as a baseline to validate the learning algo ')
    args = parser.parse_args()  

    args.output_dir = 'results/{}'.format(args.exp_name)
    env = setter.set_env(args) 
    policy = setter.set_policy(args,env) 
    buffer = setter.set_buffer(args,env) 
    logger = setter.set_logger(args) 
    algo = setter.set_algo(args,policy,buffer,env,logger) 

    ep = 0 
    while ep < args.train_episodes : 
        ep_stats = algo.collect_rollout()  
        logger.log(ep_stats,step=ep)
        train_stats = algo.train() 
        logger.log(train_stats,step=ep)
        ep+=1 
        logger.dump_tabular() 

    