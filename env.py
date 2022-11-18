import gym 
from gym import spaces
import numpy as np 
import random 

class Monkey : 

    def __init__(self,init_pos,id,rank) : 
        self.pos = init_pos 
        self.id = id 
        self.position_history = [init_pos]
        self.rank = rank 
        self.reward_history = [] 

    def update_pos(self,new_pos,r) : 
        self.pos = new_pos 
        self.position_history.append(new_pos) 
        self.reward_history.append(r) 

    def reset_agent(self,init_pos,rank) : 
        self.pos,self.position_history,self.reward_history,self.rank = init_pos, [init_pos], [] ,rank 


class SortingEnv(gym.Env):
    def __init__(self,n=4) :
        self.height = 5 
        self.n = n 
        if self.n%2!=0:
            raise ValueError('n should be odd for environment generation') 
        self.width = self.get_width(n) 
        if n!=2 :
            self.width -=1  
        self.action_space = [spaces.Discrete(5) for _ in range(n)] 
        self.observation_space = spaces.Tuple((spaces.Discrete(self.height),spaces.Discrete(self.width))) 
        self.moves = {0:[-1,0],1:[0,1],2:[1,0],3:[0,-1],4:[0,0]}  
        self.max_reward = n*2 
        self.min_reward = 2 
        self.invalid_move_reward = -2
        self.out_of_bounds_reward = -2 
        self.build_env() 
        self.reset() 

    def reset(self) : 
        self.fight_history = {} 
        init_pos = self.sample_agent_positions() 
        self.set_agent_positions(init_pos)
        agent_ranks = [i for i in range(1,self.n+1)]
        random.shuffle(agent_ranks) 
        self.agent_ranks = {i:agent_ranks[i-1] for i in range(1,self.n+1) }
        self.agents ={} 
        for i in range(1,self.n+1) : 
            self.agents[i] = Monkey(init_pos[i-1],i,self.agent_ranks[i-1]) 

    def step(self,actions) : 
        new_positions, rewards = self.collision_check(actions) 
       
        for monkey in self.agents : 
            monkey.update(new_positions[monkey.id],rewards[monkey.id]) 
        self.set_agent_positions(new_positions) 
        return 0,rewards,0 

    def update_agent_map(self,newpos) : 
        self.agent_map =  np.zeros((self.height,self.width))

    def add(self,pos,move) : 
        return (pos[0]+move[0],pos[1]+move[1])

    def build_env(self):
        self.world_map = np.zeros((self.height,self.width))
        self.agent_map =  np.zeros((self.height,self.width))
        self.banana_locations = [i for i in range(0,self.width,3)] 
        self.banana_rewards= {} 
        for j in [0,self.height-1] :
            for i in range(self.width) : 
                if i not in self.banana_locations:
                    self.world_map[j][i] = -1 
                else :
                    self.world_map[j][i] = 1   

        current_reward = self.max_reward 
        for i in self.banana_locations : 
            for j in [0,self.height-1] :
                self.banana_rewards[(j,i)] = current_reward 
                current_reward-=self.min_reward 
        
    def sample_random_location(self) : 
        random_height = np.random.randint(1,self.height-1) 
        random_width = np.random.randint(0,self.width) 
        return [random_height,random_width]

    def sample_agent_positions(self) : 
        pos = [] 
        for _ in range(self.n): 
            new_pos = self.sample_random_location() 
            while new_pos in pos : 
               new_pos = self.sample_random_location()  
            pos.append(new_pos)  
        return pos 

    def set_agent_positions (self,initial_positions) : 
        for i,pos in enumerate(initial_positions) : 
            self.agent_map[pos[0]][pos[1]] = i+1  
    
    def get_width(self,n) :
        if n==2 : 
            return 2 
        else :
            return 3 + self.get_width(n-2)

    def collision_check(self,actions) :         
        new_pos_dict,rewards_dict,not_set ={}, {} ,[] 
        for agent_id in range(1,self.n+1) : 
            agent_pos = self.agents[agent_id-1].pos 
            agent_move = self.moves[actions[agent_id]] 
            newpos = self.add(agent_pos,agent_move)  
            agent_flag = self.agent_map[newpos[0]][newpos[1]]
            #Out of Bounds Move 
            if newpos in self.banana_rewards : 
                not_set.append(newpos)
            elif newpos[0] <0 or newpos[0] == self.height or newpos[1] <0 or newpos[1] == self.width :
                new_pos_dict[agent_id] = agent_pos 
                rewards_dict[agent_id] = self.out_of_bounds_reward 
            # Into Obstacles
            elif self.world_map[newpos[0]][newpos[1]] == -1 : 
                new_pos_dict[agent_id] = agent_pos 
                rewards_dict[agent_id] = self.invalid_move_reward 
            # Easy to check valid move
            elif agent_flag == 0 : 
                new_pos_dict[agent_id] = newpos 
                if newpos in self.banana_rewards  : 
                    rewards_dict[agent_id] = self.banana_rewards[newpos] 
                else : 
                    rewards_dict[agent_id] = 0 
            # Trying to move into another agent 
            elif agent_flag != 0 : 
                not_set.append(agent_id)  
            else :
                print('Which case is this!?') 
            
        if len(not_set)!=0 :
            for agent_id in not_set : 
                agent_pos = self.agents[agent_id-1].pos 
                agent_move = self.moves[actions[agent_id]] 
                newpos = self.add(agent_pos,agent_move)  
                agent_flag = self.agent_map[newpos[0]][newpos[1]]
                # Easy to check valid move (In case the agent it was trying to move into changed positions)
                if agent_flag == 0 and newpos not in self.banana_rewards : 
                    new_pos_dict[agent_id] = newpos 
                    rewards_dict[agent_id] = 0 
                    not_set.remove(agent_id)
                # Trying to move into another agent 
                elif agent_flag != 0 : 
                    # That agent wants to stay at its current position 
                    if actions[agent_flag] ==0 :
                        if newpos not in self.banana_rewards : 
                            new_pos_dict[agent_id] = agent_pos 
                            rewards_dict[agent_id] = self.invalid_move_reward
                            not_set.remove(agent_id)
                        else : 
                            #FIGHT 
                            if self.agents[agent_id].rank > self.agents[agent_flag].rank : 
                                new_pos_dict[agent_id] = newpos 
                                rewards_dict[agent_id] = self.banana_rewards[newpos] 
                                new_pos_dict[agent_flag] = agent_pos 
                                rewards_dict[agent_flag] = 0 
                                not_set.remove(agent_flag) 
                                not_set.remove(agent_id)
                                self.fight_history[(agent_flag,agent_id)] = agent_id
                            else : 
                                new_pos_dict[agent_id] = agent_pos 
                                rewards_dict[agent_id] = 0 
                                new_pos_dict[agent_flag] = newpos 
                                rewards_dict[agent_flag] = self.banana_rewards[newpos]  
                                not_set.remove(agent_flag) 
                                not_set.remove(agent_id)
                                self.fight_history[(agent_flag,agent_id)] = agent_flag
                    else : 
                        # I tried to move into somewho who is also trying to move into someone, let's deal with this in final pass
                        pass   
                else :
                    print('Which case is this!?')

        #Some iterations to see if chain of collisions resolve itself, this case could arise if agents are following each other in a line
        for i in range(self.n-1) : 
            for agent_id in not_set : 
                    agent_pos = self.agents[agent_id-1].pos 
                    agent_move = self.moves[actions[agent_id]] 
                    newpos = self.add(agent_pos,agent_move)  
                    agent_flag = self.agent_map[newpos[0]][newpos[1]]
                    # Easy to check valid move (In case the agent it was trying to move into changed positions)
                    if agent_flag == 0 and newpos not in self.banana_rewards : 
                        new_pos_dict[agent_id] = newpos 
                        rewards_dict[agent_id] = 0 
                        not_set.remove(agent_id)

        # There must be a cycle of some sort, let's stop everyone and give 0 reward 
        for agent_id in not_set : 
                    new_pos_dict[agent_id]  = self.agents[agent_id-1].pos 
                    rewards_dict[agent_id] = 0 
        assert len(not_set) ==0 
        return new_pos_dict,rewards_dict

if __name__=='__main__':
    env = SortingEnv(4) 