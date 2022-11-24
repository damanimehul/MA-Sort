from typing import Dict, Iterable, List, Optional, Tuple, Union 
import torch as th 
import numpy as np 
import random 

def obs_as_tensor(
    obs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], device: th.device
) -> Union[th.Tensor, TensorDict]:
    """
    Moves the observation to the given device.
    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs).to(device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")

def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.
    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False