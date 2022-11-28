import torch as th 
from torch import nn 
from utils import * 


class ActorCriticPolicy(nn.Module):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(self,observation_space,action_space, lr=5e-4,net_arch= None,activation_fn = nn.ReLU,normalize_images = True, optimizer_class = th.optim.Adam,optimizer_kwargs = None,device:Union[th.device, str] = "auto"):

        super(ActorCriticPolicy, self).__init__()
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5 
        self.observation_space = observation_space
        self.action_space = action_space
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr 
        # Default network architecture, from stable-baselines
        if net_arch is None:
                net_arch = [128,128,dict(pi=[64], vf=[64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images
        self.features_dim = observation_space['features']
        self.device= get_device(device)
       
        # Action distribution
        self.action_dist = CategoricalDistribution(action_space.n)

        self._build()

    
    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self):
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()
        pi = self.mlp_extractor.latent_dim_pi
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1) 
        self.optimizer = self.optimizer_class(self.parameters(),lr=self.lr,**self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self.mlp_extractor(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) :
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=mean_actions)
       
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.
        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed 
        pi, vf = self.mlp_extractor(obs)
        distribution = self._get_action_dist_from_latent(pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(vf)
        entropy = distribution.entropy() 
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor) : 
        """
        Get the current policy distribution given the observations.
        :param obs:
        :return: the action distribution.
        """
        pi = self.mlp_extractor.forward_actor(obs)
        return self._get_action_dist_from_latent(pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        :param obs:
        :return: the estimated values.
        """
        vf = self.mlp_extractor.forward_critic(obs)
        return self.value_net(vf)

if __name__=='__main__' : 
    from sorting_env.env  import SortingEnv
    env = SortingEnv() 
    policy = ActorCriticPolicy(env.observation_shape,env.action_space) 
    obs = env.reset() 
    for _ in range(100) : 
        actions = {} 
        for id in range(1,5) :          
            o =  th.tensor(obs[id],dtype=th.float32)
            a = policy._predict(o)  
            actions[id] = int(a)
        env.step(actions) 