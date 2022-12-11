from sorting_env.env import SortingEnv
from A2C import A2C
from buffers import * 
from policies import * 
from logger import * 
from PPO import PPO 

def set_env(args) : 
    return SortingEnv(n=args.n,obs_type=args.obs_type,random_init=args.random_init,
    fights_info=args.fights_info,shuffle_ranks=args.shuffle_ranks,norm_reward=args.norm_rewards,memory=args.memory)

def set_algo(args,policy,buffer,env,logger) : 
    gamma = args.gamma 
    multi_agent = not args.single_agent 
    max_ep_len = args.max_ep_len 
    save_gifs = args.save_gifs
    gif_frequency = args.gif_frequency  
    gif_path = logger.gif_directory 
    v_coeff = args.v_coeff 
    ent_coeff = args.ent_coeff 
    if args.algo =='a2c' : 
        return A2C(policy=policy,buffer=buffer,env=env,gamma=gamma,multi_agent=multi_agent,max_ep_len=max_ep_len,
    save_gifs=save_gifs,gif_frequency=gif_frequency,gif_path=gif_path,vf_coef=v_coeff,multi_policy=args.multi_policy,ent_coef=ent_coeff) 
    elif args.algo == 'ppo' : 
        return PPO(policy=policy,buffer=buffer,env=env,gamma=gamma,multi_agent=multi_agent,max_ep_len=max_ep_len,
    save_gifs=save_gifs,gif_frequency=gif_frequency,gif_path=gif_path,vf_coef=v_coeff,multi_policy=args.multi_policy) 
    else :
        raise NotImplementedError 

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
        num_eps = args.train_freq 
        return MultiAgentBuffer(obs_dim=obs_dim,action_dim=action_dim,num_agents=num_agents,gamma=gamma,buffer_size=buffer_size,num_eps=num_eps) 
    else : 
        raise NotImplementedError 

def set_policy(args,env) : 
    if 'features' in env.observation_shape :
        lr = args.lr 
        observation_space = env.observation_shape
        action_space = env.action_space 
        net_arch = args.hidden_sizes  
        if args.multi_policy : 
            policy = {id : ActorCriticPolicy(observation_space=observation_space,action_space=action_space,lr=lr,net_arch=net_arch) for id in range(1,args.n+1) }
            return policy
        else :
            return ActorCriticPolicy(observation_space=observation_space,action_space=action_space,lr=lr,net_arch=net_arch) 
    else :
        raise NotImplementedError

def set_logger(args) : 
    return EpochLogger(args) 