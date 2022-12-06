import gym 
from gym import spaces
import numpy as np 
import random 
import matplotlib.pyplot as plt
from PIL import Image

class Observer: 
    def __init__(self,env,obs_type='grid',fights_info=True) :
        self.env = env 
        self.obs_type = obs_type  
        self.fights_info = fights_info 
        self.observation_shape = self.obs_shape() 

    def obs_shape(self) : 
        if self.obs_type=='grid': 
            size = max(self.env.height,self.env.width)
            return {'grid':[size,size,3]} 
        elif self.obs_type == 'features' : 
            # shape = all agent positions + banana locations + my position + my id 
            shape = self.env.n*2 + self.env.n*2 + 2 + self.env.n*3+ 1 
            if self.fights_info : 
                shape +=2 
            return {'features':shape} 
        elif self.obs_type == 'both' : 
            size = max(self.env.height,self.env.width)
            shape = self.env.n*2 + self.env.n*2 + self.env*2 + self.env.n 
            if self.fights_info : 
                shape +=2 
            return {'grid': [size,size,3], 'features':shape} 

    def get_obs(self) : 
        if self.obs_type == 'grid' : 
            return self.grid_obs() 
        elif self.obs_type == 'features': 
            return self.feature_obs() 
        elif self.obs_type =='both' : 
            grid_obs = self.grid_obs()
            feature_obs = self.feature_obs() 
            obs = {id: {'grid':grid_obs[id],'features':feature_obs[id]} for id in range(1,self.env.n+1)} 
            return obs 

    def grid_obs(self): 
        size = max(self.env.height,self.env.width)
        obs = -1* np.ones((size,size,3)) 
        obs_dict = {} 
        for id in range(1,self.env.n+1) : 
            obs = -1* np.ones((size,size,3))
            obs[:,:,2] = np.zeros((size,size))
            for i in range(self.env.height) :
                for j in range(self.env.width) : 
                    obs[i,j,0] = self.env.world_map[i,j] 
                    obs[i,j,1] = self.env.agent_map.map[i,j]
                    if self.env.agent_map.map[i,j] ==id :
                        obs[i,j,2] = 1
            obs_dict[id] = obs 
        return obs_dict

    def feature_obs(self) : 
        all_pos = [] 
        for id in range(1,self.env.n+1) : 
            agent = self.env.agents[id] 
            pos = list(agent.pos) 
            all_pos.append(pos[0]/self.env.height) 
            all_pos.append(pos[1]/self.env.width)
        reward_pos = [] 
        for k in self.env.banana_rewards : 
            reward_pos.append(k[0]/self.env.height) 
            reward_pos.append(k[1]/self.env.height)
        all_pos,reward_pos = np.array(all_pos),np.array(reward_pos) 
        obs_dict = {} 
        for id in range(1,self.env.n+1) : 
            agent = self.env.agents[id] 
            pos = list(agent.pos) 
            pos[0],pos[1] = agent.pos[0]/self.env.height, agent.pos[1] / self.env.width
            agent_id = agent.id 
            one_hot_id = np.zeros((self.env.n)) 
            one_hot_id[agent_id-1] = 1 
            fought = agent.fight_history[-1] 
            obs = np.concatenate([all_pos,reward_pos,pos,one_hot_id,one_hot_id,one_hot_id,[fought]])
            if self.fights_info : 
                total_fights = agent.fights 
                fights_won = agent.fights_won 
                if total_fights !=0 :
                    ratio = fights_won/total_fights
                else :
                    ratio = 0
                obs = np.concatenate([obs,[ratio,min(total_fights,self.env.n)]])
            obs_dict[id] = obs 
        return obs_dict
     
    def fixed_feature_obs(self,agent_pos,agent_id=1) : 
        all_pos,reward_pos = [] , []
        for id in range(1,self.env.n+1) : 
            if id !=agent_id : 
                agent = self.env.agents[id] 
                pos = list(agent.position_history[0]) 
                all_pos.append(pos[0]/self.env.height) 
                all_pos.append(pos[1]/self.env.width)
            else :
                pos = agent_pos 
                all_pos.append(pos[0]/self.env.height) 
                all_pos.append(pos[1]/self.env.width)
        for k in self.env.banana_rewards : 
            reward_pos.append(k[0]/self.env.height) 
            reward_pos.append(k[1]/self.env.height)
        all_pos,reward_pos = np.array(all_pos),np.array(reward_pos) 
        obs_dict = {}  
        pos = list(agent_pos) 
        pos[0],pos[1] = agent.pos[0]/self.env.height, agent.pos[1] / self.env.width
        one_hot_id = np.zeros((self.env.n)) 
        one_hot_id[agent_id-1] = 1 
        obs = np.concatenate([all_pos,reward_pos,pos,one_hot_id,one_hot_id,one_hot_id,[agent.fight_history[-1]]])
        if self.fights_info : 
            obs = np.concatenate([obs,[0,0]])
        obs_dict[agent_id] = obs 
        return obs_dict