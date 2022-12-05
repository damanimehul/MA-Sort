import numpy as np 
from utils import * 
class MultiAgentBuffer() : 
    # Wraps around single agent buffer 
    def __init__(self,num_agents=4,buffer_size=100,action_dim=5,obs_dim=24,device:Union[th.device, str] = "auto",gae_lambda= 1,gamma=0.99) : 
        self.num_agents = num_agents 
        self.device= get_device(device)
        self.buffers = {id:RolloutBuffer(buffer_size,action_dim,obs_dim,device,gae_lambda,gamma) for id in range(1,self.num_agents+1)}
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

    def sample_batch(self,id=None) : 
        if id is None : 
            id = np.random.randint(1,self.num_agents+1) 
        return self.buffers[id].sample_batch() 

class RolloutBuffer():
    def __init__(self,buffer_size=100,action_dim=5,obs_dim=22,device:Union[th.device, str] = "auto",gae_lambda= 1,gamma=0.99):

        self.buffer_size = buffer_size 
        self.action_dim = action_dim 
        self.obs_dim = obs_dim 
        self.device= get_device(device) 
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.pos = 0 
        self.full = False 
        self.observations = np.zeros((self.buffer_size,self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size), dtype=np.float32)
        self.values = np.zeros((self.buffer_size), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size), dtype=np.float32)
        self.generator_ready = False

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
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

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
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def sample_batch(self,batch_size=None) : 
        indices = np.random.permutation(self.buffer_size)
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = min(self.buffer_size,self.pos) 
        samples =  self._get_samples(indices[0 :batch_size])
        #for key in samples : 
        #    samples[key] = obs_as_tensor(samples[key],self.device)
        return samples 

    def _get_samples(self, batch_inds):
        data = dict(
            obs=self.observations[batch_inds],
            actions=th.tensor(self.actions[batch_inds]),
            values=self.values[batch_inds].flatten(),
            log_probs=self.log_probs[batch_inds].flatten(),
            advantages=th.tensor(self.advantages[batch_inds].flatten()),
            returns=th.tensor(self.returns[batch_inds].flatten()),
        )
        return data 

   