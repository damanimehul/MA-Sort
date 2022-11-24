import argparse 

if __name__ =='__main__' : 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--obs_type',type=str,default='features',help='Select the observation type between grid/features/both')
    parser.add_argument('--n',type=int,default=4,help='Number of agents to make the environment from')
    parser.add_argument('--random_init',type=int,default=True,help='Agent positions are sampled randomly on the grid') 
    parser.add_argument('--save_gifs',type=int,default=True,help='GIFS are saved periodically every gif_frequency steps') 
    parser.add_argument('--gif_frequency',type=int,default=50,help='New gif every 50 episodes')
    parser.add_argument('--gamma',type=float,default=0.995,help='Discount Factor')
    parser.add_argument('--train_episodes',type=int,default=10000,help='Number of episodes to train for')
    parser.add_argument('--wandb',action='store_true',default=False,help='Log on wandb')
    args = parser.parse_args()  

    policy = setter.set_policy(args) 
    buffer = setter.set_buffer(args) 
    algo = setter.set_algo(args,policy,buffer) 
    logger = setter.set_logger(args) 

    ep = 0 
    while ep < args.train_episodes : 
        ep_ret,ep_stats = algo.collect_rollout()  
        train_stats = algo.train() 
        logger.log(ep_stats,train_stats) 
        ep+=1 

    