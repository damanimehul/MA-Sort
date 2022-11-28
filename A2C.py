import torch as th
from torch.nn import functional as F
from utils import * 
from typing import Dict, Iterable, List, Optional, Tuple, Union 
from PIL import Image

class A2C():
   
    def __init__(self,policy, env, buffer, gamma = 0.99, gae_lambda= 1.0, ent_coef = 0.0, vf_coef = 0.5, max_grad_norm = 0.5, seed = 0,num_iterations=10, 
    device:Union[th.device, str] = "auto",multi_agent=True,max_ep_len=100,save_gifs=False,gif_frequency=50,gif_path=None):

        self.policy = policy 
        self.env = env 
        self.buffer = buffer 
        self.gamma = gamma 
        self.gae_lambda = gae_lambda
        self.seed = seed 
        self.device= get_device(device)
        self.ent_coef = ent_coef 
        self.vf_coef = vf_coef 
        self.max_grad_norm = max_grad_norm
        self.seed = seed 
        self.num_iterations = min(num_iterations,env.n)
        self.multi_agent = multi_agent 
        self.max_ep_len = max_ep_len
        self.num_agents = env.n
        self.save_gifs = save_gifs 
        self.gif_frequency = gif_frequency
        self.set_random_seed(seed) 
        self.policy = self.policy.to(self.device)
        self.episode_num = 0 
        self.gif_path = gif_path
        if save_gifs : 
            assert gif_path is not None 
 
    def set_random_seed(self, seed):
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)
        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)

    def train(self):
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
     
        for _ in range(self.num_iterations) : 
            batch = self.buffer.sample_batch() 
            obs,actions,values, log_probs,advantages,returns = batch['obs'],batch['actions'],batch['values'],batch['log_probs'],batch['advantages'],batch['returns']
            obs = obs_as_tensor(obs, self.device)
            values, log_prob, entropy = self.policy.forward(obs) 
            policy_loss = -(advantages * log_prob).mean() 
            value_loss = F.mse_loss(returns, values) 
            entropy_loss = -th.mean(entropy)
            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
       
        train_stats = {} 
        train_stats['Policy Loss'] = policy_loss.item() 
        train_stats['Value Loss'] = value_loss.item() 
        train_stats['Entropy Loss'] = entropy_loss.item() 
        return train_stats

    def collect_rollout(self) : 
        if self.multi_agent : 
            return self.collect_multi_rollout() 
        else : 
            raise NotImplementedError
        
    def collect_multi_rollout(self) :
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """ 

        def combined_reward(rewards) : 
            r = 0 
            for k,v in rewards.items() : 
                r +=v 
            return r 

        save_gif = False 
        if self.save_gifs and self.episode_num%self.gif_frequency ==0 :
            save_gif = True 

        ep_ret = 0 
        obs = self.env.reset()
        t = 0
        self.buffer.reset() 

        if save_gif : 
            array = self.env.render() 
            imgs =[Image.fromarray(array)]  

        while t < self.max_ep_len : 
            actions,env_actions,values,log_probs = {}, {} , {} , {} 
            with th.no_grad():
                    for id in range(1,self.num_agents+1) : 
                # Convert to pytorch tensor or to TensorDict
                        obs_tensor = obs_as_tensor(obs[id], self.device)
                        actions[id], values[id], log_probs[id] = self.policy(obs_tensor)
                        env_actions[id] = int(actions[id])
                        actions[id] = actions[id].cpu().numpy()
            new_obs, rewards, dones, infos = self.env.step(env_actions) 
            ep_ret += combined_reward(rewards)
            if save_gif : 
                array = self.env.render() 
                imgs.append(Image.fromarray(array)) 
            t+=1 
            self.buffer.add(obs, actions, rewards,values, log_probs)

        terminal_values = {} 
        with th.no_grad():
            for id in range(1,self.num_agents+1) : 
                    obs_tensor = obs_as_tensor(new_obs[id], self.device)
                    terminal_values[id] = self.policy.predict_values(obs_tensor)[0]
           
        self.buffer.compute_returns_and_advantages(last_values=terminal_values)
        self.episode_num +=1 
        if save_gif : 
            imgs[0].save(self.gif_path+"{}.gif".format(self.episode_num), save_all=True, append_images=imgs[1:], duration=10, loop=0)

        stats = self.env.get_stats() 
        stats['Returns'] = ep_ret
        stats['Episodes'] = self.episode_num
        return stats 

if __name__=='__main__' : 
    from sorting_env.env  import SortingEnv
    from policies import ActorCriticPolicy
    from buffers import MultiAgentBuffer
    env = SortingEnv() 
    policy = ActorCriticPolicy(env.observation_shape,env.action_space) 
    buffer = MultiAgentBuffer() 
    algo = A2C(policy,env,buffer)
    obs = env.reset() 
    for _ in range(100) : 
        algo.collect_multi_rollout() 
        algo.train()