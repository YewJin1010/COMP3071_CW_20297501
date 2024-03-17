import torch
import numpy as np
from envs.lunar_lander import LunarLander

def initialise_seeds(train_env, test_env):
    SEED = 1234

    train_env.seed(SEED)
    test_env.seed(SEED+1)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def initialise_lunar_lander():
    train_env = LunarLander()
    test_env = LunarLander()

    initialise_seeds(train_env, test_env)
    
    LunarLander.gravity = -12
    LunarLander.enable_wind = True
    LunarLander.wind_power = 20
    LunarLander.turbulence_power = 2    

    return train_env, test_env

