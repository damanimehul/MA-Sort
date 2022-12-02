from sorting_env.env import SortingEnv
from A2C import A2C
from buffers import * 
from policies import * 
from logger import * 

def set_env(args) : 
    return SortingEnv(n=args.n,obs_type=args.obs_type,random_init=args.random_init,fights_info=args.fights_info,shuffle_ranks=args.shuffle_ranks)

def set_algo(args,policy,buffer,env,logger) : 
    gamma = args.gamma 
    multi_agent = not args.single_agent 
    max_ep_len = args.max_ep_len 
    save_gifs = args.save_gifs
    gif_frequency = args.gif_frequency  
    gif_path = logger.gif_directory 
    v_coeff = args.v_coeff
    return A2C(policy=policy,buffer=buffer,env=env,gamma=gamma,multi_agent=multi_agent,max_ep_len=max_ep_len,
    save_gifs=save_gifs,gif_frequency=gif_frequency,gif_path=gif_path,vf_coef=v_coeff)

def set_buffer(args,env) :  
    if not args.single_agent : 
        buffer_size = args.max_ep_len 
        action_dim = env.action_space.n 
        if 'features' in env.observation_shape : 
            obs_dim = env.observation_shape['features'] 
        else :
            obs_dim = env.observation_shape['grid'] 
        num_agents = args.n 
        gamma = args.gamma 
        return MultiAgentBuffer(obs_dim=obs_dim,action_dim=action_dim,num_agents=num_agents,gamma=gamma,buffer_size=buffer_size) 
    else : 
        raise NotImplementedError 

def set_policy(args,env) : 
    if 'features' in env.observation_shape :
        lr = args.lr 
        observation_space = env.observation_shape
        action_space = env.action_space 
        net_arch = args.hidden_sizes 
        return ActorCriticPolicy(observation_space=observation_space,action_space=action_space,lr=lr,net_arch=net_arch) 
    else :
        raise NotImplementedError

def set_logger(args) : 
    return EpochLogger(args) 