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
        
    LunarLander.gravity = -10.0
    LunarLander.enable_wind = False
    LunarLander.wind_power = 15.0
    LunarLander.turbulence_power = 1.5    

    return train_env, test_env

