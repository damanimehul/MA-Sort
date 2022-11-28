from sorting_env.env import SortingEnv
from A2C import A2C
from buffers import * 

def set_env(args) : 
    return SortingEnv(n=args.n,obs_type=args.obs_type,random_init=args.random_init)

def set_algo(args,policy,buffer,env) : 
    gamma = args.gamma 
    multi_agent = not args.single_agent 
    return A2C(policy=policy,buffer=buffer,env=env,gamma=gamma,multi_agent=multi_agent)

def set_buffer(args,env) :  
    if not args.single_agent : 
        return MultiAgentBuffer() 

def set_policy(args) : 
    return None 

def set_logger(args) : 
    return None 