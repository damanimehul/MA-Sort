import gym 
from gym import spaces
import numpy as np 
import random 
import matplotlib.pyplot as plt
from PIL import Image

class Monkey : 

    def __init__(self,init_pos,id,rank) : 
        self.pos = init_pos 
        self.id = id 
        self.position_history = [init_pos]
        self.rank = rank 
        self.reward_history = [] 

    def update(self,new_pos,r) : 
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
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.height),spaces.Discrete(self.width))) 
        self.moves = {0:[-1,0],1:[0,1],2:[1,0],3:[0,-1],4:[0,0]}  
        self.max_reward = n*2 
        self.min_reward = 2 
        self.invalid_move_reward = -2
        self.out_of_bounds_reward = -2 
        self.color_mapping = {1:[255,255,0],-1:[0,0,0],0:[255,255,255]} 
        self.agent_colors =  {1:[102,0,0],2:[204,0,0],3:[255,102,102],4:[0,102,102],5:[0,204,204],6:[102,255,255],7:[0,255,0],8:[155,253,155]}
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
        self.agent_color_mapping = {}  
        for i in range(1,self.n+1) : 
            self.agents[i] = Monkey(init_pos[i],i,self.agent_ranks[i]) 
            self.agent_color_mapping[i] = self.agent_colors[agent_ranks[i-1]]

    def step(self,actions) : 
        new_positions, rewards = self.collision_check(actions) 
       
        for monkey in self.agents.values() : 
            monkey.update(new_positions[monkey.id],rewards[monkey.id]) 
        self.set_agent_positions(new_positions)
        print(rewards) 
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
        pos = {} 
        for i in range(1,self.n+1): 
            new_pos = self.sample_random_location() 
            while new_pos in list(pos.values()) : 
               new_pos = self.sample_random_location()  
            pos[i] = new_pos  
        return pos 

    def set_agent_positions (self,positions) : 
        self.agent_map =  np.zeros((self.height,self.width))
        for id,pos in positions.items() :
            self.agent_map[pos[0]][pos[1]] = id

    def update_position(self,id,newpos,oldpos) : 
        self.agent_map[newpos[0]][newpos[1]] = id 
        self.agent_map[oldpos[0]][oldpos[1]] = 0

    def swap_positions(self,ids,newpos) : 
        for i,pos in zip(ids,newpos) : 
            self.agent_map[pos[0]][pos[1]] = i
    
    def get_width(self,n) :
        if n==2 : 
            return 2 
        else :
            return 3 + self.get_width(n-2)

    def invalid_action(self,pos,action) : 
        # Only checking for out of bounds or into other obstacles
        agent_move = self.moves[action] 
        newpos = self.add(pos,agent_move)  
        if newpos[0] <0 or newpos[0] == self.height or newpos[1] <0 or newpos[1] == self.width or self.world_map[newpos[0]][newpos[1]] == -1 :
            return True 
        return False 

    def collision_check(self,actions) :         
        new_pos_dict,rewards_dict,not_set ={},{},[] 
        for id in range(1,self.n+1) : 
            rewards_dict[id] = 0 
        for agent_id in range(1,self.n+1) : 
            agent_pos = self.agents[agent_id].pos 
            agent_move = self.moves[actions[agent_id]] 
            newpos = self.add(agent_pos,agent_move)  
            if tuple(agent_pos) in self.banana_rewards : 
                    not_set.append(agent_id)
            #Out of Bounds Move 
            elif newpos[0] <0 or newpos[0] == self.height or newpos[1] <0 or newpos[1] == self.width :
                new_pos_dict[agent_id] = agent_pos 
                rewards_dict[agent_id] = self.out_of_bounds_reward 
            # Into Obstacles
            elif self.world_map[newpos[0]][newpos[1]] == -1 : 
                new_pos_dict[agent_id] = agent_pos 
                rewards_dict[agent_id] = self.invalid_move_reward 
            # Easy to check valid move
            elif self.agent_map[newpos[0]][newpos[1]] == 0 : 
                new_pos_dict[agent_id] = newpos 
                if newpos in self.banana_rewards  : 
                    rewards_dict[agent_id] = self.banana_rewards[newpos] 
                else : 
                    rewards_dict[agent_id] = 0 
                self.update_position(agent_id,newpos,agent_pos)
            # Trying to move into another agent 
            elif self.agent_map[newpos[0]][newpos[1]] != 0 : 
                not_set.append(agent_id)  
            else :
                print('Which case is this 1!?') 

        remove =[] 

        if len(not_set)!=0 :
            for agent_id in not_set : 
                agent_pos = self.agents[agent_id].pos 
                agent_move = self.moves[actions[agent_id]] 
                newpos = self.add(agent_pos,agent_move)  
                # This means I am at a reward state and took an invalid action, cannot say anything about my next position as I might be displaced, but will definitely receive an invalid action reward 
                if self.invalid_action(agent_pos,actions[agent_id]) : 
                    rewards_dict[agent_id] += self.invalid_move_reward
                    # I might have been displaced, if I haven't been displaced yet, set the current position to stay (displacing will overwrite this)
                    if agent_id not in new_pos_dict :
                        new_pos_dict[agent_id] = agent_pos 
                        remove.append(agent_id)
                    continue 
                # Easy to check valid move (In case the agent it was trying to move into changed positions)
                elif self.agent_map[newpos[0]][newpos[1]] == 0 : 
                    new_pos_dict[agent_id] = newpos 
                    rewards_dict[agent_id] = 0 
                    if newpos in self.banana_rewards  : 
                        rewards_dict[agent_id] = self.banana_rewards[newpos] 
                    self.update_position(agent_id,newpos,agent_pos)
                    remove.append(agent_id)
                # Trying to move into another agent 
                elif self.agent_map[newpos[0]][newpos[1]] != 0 : 
                    agent_flag= self.agent_map[newpos[0]][newpos[1]]
                    # That agent wants to/ will stay at its current position 
                    if actions[agent_flag] ==0 or self.invalid_action(newpos,actions[agent_flag]) :
                        if newpos not in self.banana_rewards : 
                            new_pos_dict[agent_id] = agent_pos 
                            rewards_dict[agent_id] = 0  
                            remove.append(agent_id)
                        else : 
                            #FIGHT 
                            if self.agents[agent_id].rank < self.agents[agent_flag].rank : 
                                new_pos_dict[agent_id] = newpos 
                                rewards_dict[agent_id] += self.banana_rewards[newpos] 
                                new_pos_dict[agent_flag] = agent_pos 
                                remove.append(agent_flag) 
                                remove.append(agent_id)
                                self.fight_history[(agent_flag,agent_id)] = agent_id
                                self.swap_positions([agent_id,agent_flag],[newpos,agent_pos])
                            else : 
                                new_pos_dict[agent_id] = agent_pos 
                                new_pos_dict[agent_flag] = newpos 
                                rewards_dict[agent_flag] += self.banana_rewards[newpos]  
                                remove.append(agent_flag) 
                                remove.append(agent_id)
                                self.fight_history[(agent_flag,agent_id)] = agent_flag
                    else : 
                        # I tried to move into somewho who is also trying to move into someone, let's deal with this in final pass
                        pass   
                else :
                    print('Which case is this 2!?') 

        remove = list(set(remove))
        for id in remove : 
            not_set.remove(id) 
        remove =[] 
        #Some iterations to see if chain of collisions resolve itself, this case could arise if agents are following each other in a line
        for i in range(self.n-1) : 
            for agent_id in not_set : 
                if agent_id not in new_pos_dict :
                    agent_pos = self.agents[agent_id].pos 
                    agent_move = self.moves[actions[agent_id]] 
                    newpos = self.add(agent_pos,agent_move)  
                    agent_flag = self.agent_map[newpos[0]][newpos[1]]
                    # Easy to check valid move (In case the agent it was trying to move into changed positions)
                    if agent_flag == 0 and newpos not in self.banana_rewards: 
                        new_pos_dict[agent_id] = newpos 
                        rewards_dict[agent_id] = 0 
                        self.update_position(agent_id,newpos,agent_pos)
                        remove.append(agent_id)
                else :
                    remove.append(agent_id)

        remove = list(set(remove)) 
        
        for id in remove : 
            not_set.remove(id) 
        remove =[] 
        # There must be a cycle of some sort, let's stop remaining agents and give 0 reward 
        print(not_set)
        for agent_id in not_set : 
                if agent_id not in new_pos_dict : 
                    new_pos_dict[agent_id]  = self.agents[agent_id].pos 
                    rewards_dict[agent_id] = 0 
                remove.append(agent_id)
        for id in remove : 
            not_set.remove(id) 

        for id,r in rewards_dict.items() :
            if tuple(new_pos_dict[id]) in self.banana_rewards and r<0 :
                rewards_dict[id]+= self.banana_rewards[new_pos_dict[id]]

        assert len(not_set) ==0 
        return new_pos_dict,rewards_dict

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError()

        board_h, board_w = self.height,self.width
        rgb_array = np.zeros((board_h * 32, board_w * 32, 3), np.uint8) 


        for i in range(board_h):
            for j in range(board_w):
                color = self.color_mapping[int(self.world_map[i, j])]
                rgb_array[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32] = color

        for i in range(board_h):
            for j in range(board_w):
                agent_present = self.agent_map[i, j] 
                if agent_present : 
                    color = self.agent_color_mapping[int(self.agent_map[i, j])]
                    rgb_array[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32] = color

        for i in range(board_h):
            rgb_array[i * 32, :] = 0
            rgb_array[(i + 1) * 32 - 1, :] = 0
        for j in range(board_w):
            rgb_array[:, j * 32] = 0
            rgb_array[:, (j + 1) * 32 - 1] = 0

        return rgb_array

if __name__=='__main__':
    env = SortingEnv(8) 
    array = env.render()
   # plt.imshow(array) 
   # plt.show() 
   # plt.close() 
    break_flag = 0
    for j in range(1000): 
        imgs =[] 
        for _ in range(20) : 
            actions = {} 
            for i in range(1,9) : 
                actions[i] = env.action_space.sample() #int(in_actions[i-1]) 
           # try :
            env.step(actions) 
            #except :
            #    break_flag = 1
            #    break 
            array = env.render()
            imgs.append(Image.fromarray(array))
        print(j) 
        imgs[0].save("array2.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)
        if break_flag :
            break 
        env.reset() 
      #  plt.imshow(array) 
      #  plt.show()
      #  plt.close() 

