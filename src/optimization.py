'''
Optimization Methods will have methods for parameter optimization 
'''
import numpy as np
from src.model import simulate_model

def compute_error(params: tuple, t_obs: np.ndarray, init_cond: list, x_obs: np.ndarray, y_obs: np.ndarray)-> float:
    '''
    Compute error between observed and simulated population trajectories 

    Args:
    params (tuple): Model parameters (alpha, beta, gamma, delta)
    '''
    simulate = simulate_model(params, t_obs, init_cond)  
    x_error = np.mean((simulate[0] - x_obs)**2)
    y_error = np.mean((simulate[1] - y_obs)**2)
    return x_error, y_error

def local_optimization():
    return 0

def global_optimization():
    return 0
