import torch as th
from torch.nn import functional as F
from utils import * 
from typing import Dict, Iterable, List, Optional, Tuple, Union 

class A2C():
   
    def __init__(self,policy, env, buffer, gamma = 0.99, gae_lambda= 1.0, ent_coef = 0.0, vf_coef = 0.5, max_grad_norm = 0.5, seed = 0,num_iterations=10, 
    device:Union[th.device, str] = "auto",multi_agent=True,max_ep_len=100):

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
        self.num_iterations = num_iterations  
        self.multi_agent = multi_agent 
        self.max_ep_len = max_ep_len
        self.num_agents = env.n
        self.set_random_seed(seed) 
        self.policy = self.policy.to(self.device)
 
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
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)

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
       
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item()) 

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
        obs = self.env.reset()
        t = 0
        self.buffer.reset()

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
            t+=1 
            self.buffer.add(obs, actions, rewards,values, log_probs)

        terminal_values = {} 
        with th.no_grad():
            for id in range(1,self.num_agents+1) : 
                    obs_tensor = obs_as_tensor(new_obs[id], self.device)
                    terminal_values[id] = self.policy.predict_values(obs_tensor)[0]
           
        self.buffer.compute_returns_and_advantages(last_values=terminal_values)
    
        return True

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