'''
Optimization Methods will have methods for parameter optimization 
'''
import numpy as np
from src.model import simulate_model

def compute_combined_rmse(params: tuple, t_obs: np.ndarray, init_cond: list, x_obs: np.ndarray, y_obs: np.ndarray)-> float:
    '''
    Compute error between observed and simulated population trajectories 

    Args:
    params (tuple): Model parameters (alpha, beta, gamma, delta)
    '''
    simulate = simulate_model(params, t_obs, init_cond)  
    
    x_error = (simulate[0] - x_obs)
    y_error = (simulate[1] - y_obs)
    
    total_squared_error= np.sum((x_error)**2+(y_error)**2)
    combined_rmse = np.sqrt(total_squared_error/ len(t_obs))
    
    return combined_rmse

def compute_mape(params, t_obs, init_cond, x_obs, y_obs):
    '''
    Compute mean absolute percentage error 
    '''
    simulate = simulate_model(params, t_obs, init_cond)
    
    x_mape = np.mean(np.abs((x_obs - simulate[0])/ x_obs)) * 100
    y_mape = np.mean(np.abs((y_obs - simulate[1])/ y_obs)) * 100

    return (x_mape+ y_mape)/2

def local_optimization(init_guess, bounds, t_obs, init_cond, x_obs, y_obs):
    '''
    Function to perform local optimization havent figured this out yet
    '''

    return 0

def global_optimization():
    '''
    Global optimization (Simulated annealing to find out optimized parameters)
    '''
    return 0
