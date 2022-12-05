import gym 
from gym import spaces
import numpy as np 
import random 
import matplotlib.pyplot as plt
from PIL import Image
#Python program to print topological sorting of a DAG
import networkx as nx 

class Monkey : 

    def __init__(self,init_pos,id,rank) : 
        self.pos = init_pos 
        self.id = id 
        self.position_history = [init_pos]
        self.rank = rank 
        self.reward_history = []
        self.fights = 0 
        self.fights_won = 0  
        self.had_fight = 0 
        self.fight_dict = {} 
        self.fight_history = [0]
        
    def update(self,new_pos,r,had_fight) : 
        self.pos = new_pos 
        self.position_history.append(new_pos) 
        self.reward_history.append(r) 
        self.fight_history.append(had_fight)

    def fight_update(self,win,agent_id) : 
        self.fight_dict[agent_id] = win 
        if win :
            self.fights+=1 
            self.fights_won +=1 
        else :
            self.fights+=1 

    def reset_agent(self,init_pos,rank) : 
        self.pos,self.position_history,self.fight_history,self.reward_history,self.rank,self.fights,self.fights_won = init_pos, [init_pos],[0], [] ,rank ,0,0 
        self.fight_dict = {} 

class AgentMap : 
    def __init__(self,env) : 
        self.env = env 
        self.height = env.height 
        self.width = env.width
        self.random_init = env.random_init
        self.n = env.n 
         
    def reset_map(self) : 
        self.map =  np.zeros((self.height,self.width))

    def set_agent_positions(self,positions) : 
        self.reset_map() 
        for id,pos in positions.items() :
            self.map[pos[0]][pos[1]] = id

    def update(self,id,newpos,oldpos) : 
        self.map[newpos[0]][newpos[1]] = id 
        self.map[oldpos[0]][oldpos[1]] = 0

    def swap(self,ids,newpos) : 
        for i,pos in zip(ids,newpos) : 
            self.map[pos[0]][pos[1]] = i 

    def query(self,newpos) : 
        try :
            return self.map[newpos[0]][newpos[1]]
        except :
            print('Not Found on Map, invalid query') 

    def sample_random_location(self) : 
        sampled = False 
        while not sampled : 
            random_height = np.random.randint(0,self.height) 
            random_width = np.random.randint(0,self.width)  
            if self.env.world_map[random_height][random_width] != -1 : 
                sampled = True 
        return [random_height,random_width] 

    def sample_fixed_location(self) :
        # This samples fixed locations at the middle row, so all agents start in the middle row. 
        height = int(self.height/2)
        random_width = np.random.randint(0,self.width)  
        return [height,random_width] 

    def sample_positions(self) : 
        pos = {} 
        for i in range(1,self.n+1): 
            if self.random_init : 
                new_pos = self.sample_random_location() 
            else : 
                new_pos = self.sample_fixed_location() 
            while new_pos in list(pos.values()) : 
               if self.random_init : 
                    new_pos = self.sample_random_location() 
               else : 
                    new_pos = self.sample_fixed_location()   
            pos[i] = new_pos  
        return pos 

    def initialize_agents(self) : 
        self.reset_map() 
        pos = self.sample_positions() 
        self.set_agent_positions(pos) 
        return pos 

class FightGraph() : 
    def __init__(self,env) : 
        self.env = env 
        self.reset() 

    def update(self,win_agent,losing_agent) : 
        # If there is a directed path between these two agents, then the recent fight was redundant as the order can be determined from environment history
        a = nx.has_path(self.graph,win_agent,losing_agent)
        b = nx.has_path(self.graph,losing_agent,win_agent) 
        if a or b : 
            return True 
        else : 
            self.graph.add_edge(win_agent,losing_agent) 
            return False 
        
    def reset(self) : 
        self.graph = nx.DiGraph()  
        for i in range(1,self.env.n+1) : 
            self.graph.add_node(i)

