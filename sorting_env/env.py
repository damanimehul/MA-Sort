import gym 
from gym import spaces
import numpy as np 
import random 
import matplotlib.pyplot as plt
from PIL import Image
from sorting_env.Observers import Observer
from sorting_env.env_utils import Monkey,AgentMap,FightGraph
from utils import * 
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns

class SortingEnv(gym.Env):
    def __init__(self,n=4,random_init=False,obs_type='features',fights_info=True,
    shuffle_ranks=True,norm_reward= False) :
        self.height = 5 
        self.n = n 
        if self.n%2!=0:
            raise ValueError('n should be odd for environment generation') 
        self.width = self.get_width(n) 
        if n!=2 :
            self.width -=1  
        self.action_space = spaces.Discrete(5)

        #Fixed variables 
        self.moves = {0:[-1,0],1:[0,1],2:[1,0],3:[0,-1],4:[0,0]}  
        self.max_reward = n*2 
        self.min_reward = 2 
        self.invalid_move_reward = -1 
        self.out_of_bounds_reward = -1 

        print('Fights Info:' , fights_info)
        print('Shuffle Ranks:',shuffle_ranks) 
        print('Random Init',random_init)

        # For rendering 
        self.color_mapping = {1:[255,255,0],-1:[0,0,0],0:[255,255,255]} 
        self.agent_colors =  {1:[102,0,0],2:[204,0,0],3:[255,102,102],4:[0,102,102],5:[0,204,204],6:[102,255,255],7:[0,255,0],8:[155,253,155]}
        self.fight_graph = FightGraph(self) 

        # These might change
        self.random_init = random_init
        self.observer = Observer(self,obs_type,fights_info=fights_info)  
        self.observation_shape = self.observer.observation_shape 
        self.shuffle_ranks = shuffle_ranks
        self.norm_reward = norm_reward

        self.build_env() 
        self.reset() 

    def reset(self) : 
        self.fight_history = {} 
        self.fight_graph.reset() 
        self.solved = False 
        self.fights,self.redundant_fights = 0,0 
        self.invalid_actions = 0 
        init_pos = self.agent_map.initialize_agents() 
        agent_ranks = [i for i in range(1,self.n+1)]
        if self.shuffle_ranks : 
            random.shuffle(agent_ranks) 
        self.agent_ranks = {i:agent_ranks[i-1] for i in range(1,self.n+1) }
        self.agents ={}
        self.agent_color_mapping = {}  
        for i in range(1,self.n+1) : 
            self.agents[i] = Monkey(init_pos[i],i,self.agent_ranks[i]) 
            self.agent_color_mapping[i] = self.agent_colors[agent_ranks[i-1]]
        return self.observer.get_obs() 

    def get_stats(self) : 
        stat_dict = {}
        stat_dict['Fights'] = self.fights 
        stat_dict['Redundant Fights'] = self.redundant_fights 
        stat_dict['Solved'] = int(self.solved) *100 
        stat_dict['Invalid Actions'] = self.invalid_actions / self.n  

        for id in range(1,self.n+1) : 
            stat_dict['Agent {} returns'.format(id)] = sum(self.agents[id].reward_history)
        return stat_dict 

    def step(self,actions) : 
        new_positions, rewards,had_fight_dict = self.collision_check(actions) 
        for monkey in self.agents.values() : 
            monkey.update(new_positions[monkey.id],rewards[monkey.id],had_fight_dict[monkey.id]) 
        self.agent_map.set_agent_positions(new_positions)
        if not self.solved : 
            self.solved = self.check_solved(new_positions)
        
        #Normalizes the banana rewards given 
        if self.norm_reward : 
            for k,v in rewards.items() : 
                if v>0 :
                    rewards[k] = v/self.max_reward

        return self.observer.get_obs(),rewards,False, None 

    def add(self,pos,move) : 
        return (pos[0]+move[0],pos[1]+move[1])

    def build_env(self):
        self.world_map = np.zeros((self.height,self.width))
        self.agent_map =  AgentMap(self)
        self.banana_locations = [i for i in range(0,self.width,3)] 
        self.banana_rewards= {} 
        for j in [0,self.height-1] :
            for i in range(self.width) : 
                if i not in self.banana_locations:
                    self.world_map[j][i] = -1 
                else :
                    self.world_map[j][i] = 1  

        current_reward = self.max_reward 
        reward_to_pos = {} 
        for i in self.banana_locations : 
            for j in [0,self.height-1] :
                self.banana_rewards[(j,i)] = current_reward 
                reward_to_pos[current_reward] = (j,i)
                current_reward-=self.min_reward 
                
        self.optimal_rank_pos = {} 
        for i in range(0,self.n) : 
            opt_reward = self.max_reward - i*self.min_reward 
            self.optimal_rank_pos[i+1] = reward_to_pos[opt_reward]
       
    def get_width(self,n) :
        if n==2 : 
            return 2 
        else :
            return 3 + self.get_width(n-2) 

    def check_solved(self,new_pos) : 
        for agent_id,pos in new_pos.items() : 
            rank  = self.agents[agent_id].rank 
            if not self.optimal_rank_pos[rank] == pos  : 
                return False 
        return True 

    def invalid_action(self,pos,action) : 
        # Only checking for out of bounds or into other obstacles
        agent_move = self.moves[action] 
        newpos = self.add(pos,agent_move)  
        if newpos[0] <0 or newpos[0] == self.height or newpos[1] <0 or newpos[1] == self.width or self.world_map[newpos[0]][newpos[1]] == -1 :
            return True 
        return False 

    def collision_check(self,actions) :         
        new_pos_dict,rewards_dict,had_fight,not_set = {},{},{},[] 
        fights= [] 
        for id in range(1,self.n+1) : 
            rewards_dict[id] = 0 
            had_fight[id] = 0 
        for agent_id in range(1,self.n+1) : 
            agent_pos = self.agents[agent_id].pos 
            agent_move = self.moves[actions[agent_id]] 
            newpos = self.add(agent_pos,agent_move)  
            #Ignore agents at reward locations for now as they might have to fight 
            if tuple(agent_pos) in self.banana_rewards : 
                    not_set.append(agent_id)
            #Out of Bounds Move 
            elif newpos[0] <0 or newpos[0] == self.height or newpos[1] <0 or newpos[1] == self.width :
                new_pos_dict[agent_id] = agent_pos 
                rewards_dict[agent_id] = self.out_of_bounds_reward  
                self.invalid_actions +=1 
            # Into Obstacles
            elif self.world_map[newpos[0]][newpos[1]] == -1 : 
                new_pos_dict[agent_id] = agent_pos 
                rewards_dict[agent_id] = self.invalid_move_reward 
                self.invalid_actions +=1 
            # Easy to check valid move
            elif self.agent_map.query(newpos) == 0 : 
                new_pos_dict[agent_id] = newpos 
                if newpos in self.banana_rewards  : 
                    rewards_dict[agent_id] = self.banana_rewards[newpos] 
                else : 
                    rewards_dict[agent_id] = 0 
                self.agent_map.update(agent_id,newpos,agent_pos)
            # Stopped at current position which is not a banana position
            elif actions[agent_id] ==4 :
                new_pos_dict[agent_id] = newpos 
                rewards_dict[agent_id] = 0 
            # Trying to move into another agent 
            elif self.agent_map.query(newpos) != 0 : 
                not_set.append(agent_id)  
            else :
                print('Which case is this 1!?') 

        remove =[] 
        if len(not_set)!=0 :
            for agent_id in not_set : 
                if agent_id not in remove : 
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
                    elif self.agent_map.query(newpos) == 0 : 
                        new_pos_dict[agent_id] = newpos 
                        rewards_dict[agent_id] = 0 
                        if newpos in self.banana_rewards  : 
                            rewards_dict[agent_id] = self.banana_rewards[newpos] 
                        self.agent_map.update(agent_id,newpos,agent_pos)
                        remove.append(agent_id)
                    # Trying to move into another agent 
                    elif self.agent_map.query(newpos) != 0 : 
                        agent_flag= self.agent_map.query(newpos)
                        # I was staying stopped, might be displaced so check for that
                        if actions[agent_id] == 4:
                            if agent_id not in new_pos_dict :
                                new_pos_dict[agent_id] = agent_pos 
                                rewards_dict[agent_id] = self.banana_rewards[newpos] 
                                remove.append(agent_id)
                                continue 

                        # That agent wants to/ will stay at its current position 
                        elif actions[agent_flag] == 4 or self.invalid_action(newpos,actions[agent_flag]) or newpos in self.banana_rewards :
                            if newpos not in self.banana_rewards : 
                                new_pos_dict[agent_id] = agent_pos 
                                rewards_dict[agent_id] = 0  
                                remove.append(agent_id)
                            else : 
                                #FIGHT 
                                if self.agents[agent_id].rank < self.agents[agent_flag].rank :  
                                    f, rev_f = [agent_flag,agent_id] , [agent_id,agent_flag] 
                                    if f not in fights and rev_f not in fights: 
                                        new_pos_dict[agent_id] = newpos 
                                        new_pos_dict[agent_flag] =  agent_pos 
                                        rewards_dict[agent_id] += self.banana_rewards[newpos] 
                                        rewards_dict[agent_flag] = 0 
                                        had_fight[agent_flag] = 1 
                                        had_fight[agent_id] = 1
                                        remove.extend([agent_flag,agent_id]) 
                                        self.fight_history[(agent_flag,agent_id)] = agent_id
                                        self.agent_map.swap([agent_id,agent_flag],[newpos,agent_pos]) 
                                        self.agents[agent_id].fight_update(1,agent_flag) 
                                        self.agents[agent_flag].fight_update(0,agent_id) 
                                        redundant = self.fight_graph.update(agent_id,agent_flag) 
                                        self.fights+=1 
                                        if redundant :
                                            self.redundant_fights +=1 
                                        fights.append(f) 
                                else : 
                                    f, rev_f = [agent_flag,agent_id] , [agent_id,agent_flag] 
                                    if f not in fights and rev_f not in fights: 
                                        new_pos_dict[agent_id] = agent_pos
                                        new_pos_dict[agent_flag] =  newpos
                                        rewards_dict[agent_flag] += self.banana_rewards[newpos]
                                        rewards_dict[agent_id] = 0 
                                        had_fight[agent_flag] = 1 
                                        had_fight[agent_id] = 1
                                        remove.extend([agent_flag,agent_id])
                                        self.fight_history[(agent_flag,agent_id)] = agent_flag
                                        self.agents[agent_id].fight_update(0,agent_flag) 
                                        self.agents[agent_flag].fight_update(1,agent_id)
                                        self.fights+=1 
                                        redundant = self.fight_graph.update(agent_flag,agent_id) 
                                        if redundant :
                                            self.redundant_fights +=1 
                                        fights.append(f) 
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
        for _ in range(self.n-1) : 
            for agent_id in not_set : 
                if agent_id not in new_pos_dict :
                    agent_pos = self.agents[agent_id].pos 
                    agent_move = self.moves[actions[agent_id]] 
                    newpos = self.add(agent_pos,agent_move)  
                    agent_flag = self.agent_map.query(newpos)
                    # Easy to check valid move (In case the agent it was trying to move into changed positions)
                    if agent_flag == 0 and newpos not in self.banana_rewards: 
                        new_pos_dict[agent_id] = newpos 
                        rewards_dict[agent_id] = 0 
                        self.agent_map.update(agent_id,newpos,agent_pos)
                        remove.append(agent_id)
                else :
                    remove.append(agent_id)

            remove = list(set(remove)) 
            
            for id in remove : 
                not_set.remove(id) 
            remove =[] 
        # There must be a cycle of some sort, let's stop remaining agents and give 0 reward 
        for agent_id in not_set : 
                if agent_id not in new_pos_dict : 
                    new_pos_dict[agent_id]  = self.agents[agent_id].pos 
                    rewards_dict[agent_id] = 0 
                remove.append(agent_id)
        for id in remove : 
            not_set.remove(id) 

        for id,r in rewards_dict.items() :
            if tuple(new_pos_dict[id]) in self.banana_rewards and r<=0 :
                rewards_dict[id]+= self.banana_rewards[tuple(new_pos_dict[id])]

        #Bounding rewards as they are assigned wrongly at certain timesteps 
        for id,r in rewards_dict.items() :
            if tuple(new_pos_dict[id]) in self.banana_rewards and r> self.max_reward :
                rewards_dict[id]= int(rewards_dict[id]/2)

        assert len(not_set) ==0 
        return new_pos_dict,rewards_dict,had_fight

    def get_v_map(self,policy,device,agent_id=1) : 
        v_map = np.ones((self.height,self.width)) * 10000
        v_vals= [] 
        for i in range(self.height) : 
            for j in range(self.width) : 
                if self.world_map[i][j] !=-1 : 
                    pos =[i,j] 
                    obs = self.observer.fixed_feature_obs(pos,agent_id) 
                    obs_tensor = obs_as_tensor(obs[agent_id], device)
                    v = policy.predict_values(obs_tensor)[0]
                    v_map[i][j] = v.item() 
                    v_vals.append(v.item())

        v_min = np.min(v_vals) 
        for i in range(self.height) : 
            for j in range(self.width) : 
                if v_map[i][j] == 10000: 
                    v_map[i][j] = v_min 

        plot = sns.heatmap(v_map,linewidth=0.5)
        plt.close() 
        return plot 
     
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
                agent_present = self.agent_map.query([i, j]) 
                if agent_present : 
                    color = self.agent_color_mapping[int(self.agent_map.query([i, j]) )]
                    rgb_array[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32] = color

        for i in range(board_h):
            rgb_array[i * 32, :] = 0
            rgb_array[(i + 1) * 32 - 1, :] = 0
        for j in range(board_w):
            rgb_array[:, j * 32] = 0
            rgb_array[:, (j + 1) * 32 - 1] = 0

        # Adding this noise because gif rendering compresses identical frames which reduces episode lengths
        noise = np.zeros((board_h * 32, board_w * 32, 3), np.uint8)  
        noise[0][0][0] += np.random.randint(0,100,dtype=np.uint8)
        rgb_array += noise 

        return rgb_array

if __name__=='__main__': 
    import Observers 
    import env_utils
    env = SortingEnv(4,False) 
    map = env.reset()
    array = env.render() 
   # plt.imshow(array) 
   # plt.show() 
   # plt.close() 
    break_flag = 0
    for j in range(1): 
        imgs =[Image.fromarray(array)] 
        for _ in range(20) : 
            actions = {} 
            a = '4444'
            for i in range(1,5) : 
                actions[i] = int(a[i-1])  #env.action_space.sample()  # int(a[i-1])#
            try :
                print(actions)
                o,r,_,_ = env.step(actions) 
            except :
                break_flag = 1 
                break
            #print(r) 
           # print(o) 
            array = env.render()
            
            #plt.imshow(array) 
            #plt.show()
            #plt.close() 
            #except :
            #    break_flag = 1
            #    break 
            imgs.append(Image.fromarray(array)) 
        print(j)
        imgs[0].save("array2.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)
        if break_flag :
            break 
        env.reset() 
        array = env.render() 
        

