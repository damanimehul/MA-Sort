import numpy as np 
from utils import * 
class MultiAgentBuffer() : 
    # Wraps around single agent buffer 
    def __init__(self,num_agents=4,buffer_size=5000,action_dim=5,obs_dim=24,device:Union[th.device, str] = "auto",gae_lambda= 1,gamma=0.99,num_eps=1) : 
        self.num_agents = num_agents 
        self.device= get_device(device)
        self.buffers = {id:RolloutBuffer(buffer_size,action_dim = action_dim,obs_dim = obs_dim,device=device,gae_lambda=gae_lambda,gamma=gamma,num_eps=num_eps) for id in range(1,self.num_agents+1)}
        self.reset() 

    def reset(self) : 
        for id,buffer in self.buffers.items() : 
            buffer.reset() 
    
    def compute_returns_and_advantages(self,last_values) : 
        for id,buffer in self.buffers.items() : 
            buffer.compute_returns_and_advantage(last_values[id]) 
    
    def add(self,obs,action, reward, value,log_prob) :  
        for id,buffer in self.buffers.items() : 
            buffer.add(obs[id], action[id], reward[id],value[id], log_prob[id]) 

    def end_episode(self) : 
        for id,buffer in self.buffers.items() : 
            buffer.end_episode() 

    def sample_batch(self,id=None) : 
        if id is None : 
            id = np.random.randint(1,self.num_agents+1) 
        return self.buffers[id].sample_batch() 

class RolloutBuffer():
    def __init__(self,buffer_size=5000,action_dim=5,obs_dim=22,batch_size=256,device:Union[th.device, str] = "auto",gae_lambda= 1,gamma=0.99,num_eps=1):

        self.buffer_size = buffer_size 
        self.action_dim = action_dim 
        self.obs_dim = obs_dim 
        self.device= get_device(device) 
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.num_eps = num_eps 
        self.ep_ind = 0 
        self.batch_size = batch_size 
        self.reset()

    def reset(self) -> None:
        self.pos = 0 
        self.full = False 
        self.observations = np.zeros((self.num_eps,self.buffer_size,self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.num_eps,self.buffer_size), dtype=np.float32)
        self.rewards = np.zeros((self.num_eps,self.buffer_size), dtype=np.float32)
        self.returns = np.zeros((self.num_eps,self.buffer_size), dtype=np.float32)
        self.values = np.zeros((self.num_eps,self.buffer_size), dtype=np.float32)
        self.log_probs = np.zeros((self.num_eps,self.buffer_size), dtype=np.float32)
        self.advantages = np.zeros((self.num_eps,self.buffer_size), dtype=np.float32)
        self.ep_ind = 0 
        
    def end_episode(self) : 
        self.ep_ind += 1 
        self.pos = 0 

    def compute_returns_and_advantage(self, last_values):
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 
                next_values = last_values
            else:
                next_non_terminal = 1.0 
                next_values = self.values[self.ep_ind][step + 1]
            delta = self.rewards[self.ep_ind][step] + self.gamma * next_values * next_non_terminal - self.values[self.ep_ind][step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[self.ep_ind][step] = last_gae_lam
        self.returns[self.ep_ind] = self.advantages[self.ep_ind] + self.values[self.ep_ind]
        
    def add(self,obs,action, reward, value,log_prob) : 
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)
        self.observations[self.ep_ind][self.pos] = np.array(obs).copy()
        self.actions[self.ep_ind][self.pos] = np.array(action).copy()
        self.rewards[self.ep_ind][self.pos] = np.array(reward).copy()
        self.values[self.ep_ind][self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.ep_ind][self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def sample_batch(self,batch_size=None) : 
        indices = np.random.permutation(self.buffer_size*self.ep_ind)
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.batch_size 
        samples =  self._get_samples(indices[0 :self.batch_size])
        return samples 

    def _get_samples(self, batch_inds): 
        o = np.reshape(self.observations,(-1,self.obs_dim)) 
        a = self.actions.flatten() 
        v = self.values.flatten() 
        lp = self.log_probs.flatten() 
        adv = self.advantages.flatten() 
        ret = self.returns.flatten() 
        data = dict(
            obs=o[batch_inds],
            actions=th.tensor(a[batch_inds]),
            values=v[batch_inds].flatten(),
            log_probs=lp[batch_inds].flatten(),
            advantages=th.tensor(adv[batch_inds].flatten()),
            returns=th.tensor(ret[batch_inds].flatten()),
        )
        return data 

   