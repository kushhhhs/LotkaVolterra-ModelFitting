'''
Optimization Methods will have methods for parameter optimization 
'''
import numpy as np
from src.model import simulate_model
from scipy.optimize import minimize, dual_annealing

def compute_combined_rmse(params: tuple, t_obs: np.ndarray, init_cond: list, x_obs: np.ndarray, y_obs: np.ndarray)-> float:
    '''
    Compute error between observed and simulated population trajectories 

    Args:
    params (tuple): Model parameters (alpha, beta, gamma, delta)
    '''
    x_simulated, y_simulated = simulate_model(params, init_cond, t_obs)

    #print(t_obs.shape, x_simulated.shape, y_simulated.shape)
    
    x_error = (x_simulated - x_obs) ** 2
    y_error = (y_simulated - y_obs) ** 2
    
    total_squared_error = np.sum(x_error + y_error)
    combined_rmse = np.sqrt(total_squared_error / len(t_obs))
    
    return combined_rmse

def compute_mape(params, t_obs, init_cond, x_obs, y_obs):
    '''
    Compute mean absolute percentage error 
    '''
    x_simulated, y_simulated = simulate_model(params, init_cond, t_obs)
    
    x_mape = np.mean(np.abs((x_obs - x_simulated)/ x_obs)) * 100
    y_mape = np.mean(np.abs((y_obs - y_simulated)/ y_obs)) * 100

    return (x_mape+ y_mape)/2

def local_optimization_RMSE(init_guess, bounds, t_obs, init_cond, x_obs, y_obs):
    '''
    L-BFGS-B method to find optimized parameters by finding local minimum error (RMSE) based on initial conditions 
    '''
    result = minimize(compute_combined_rmse, init_guess, args=(t_obs, init_cond, x_obs, y_obs), bounds=bounds, method='L-BFGS-B')
    return result.x

def local_optimization_MAPE(init_guess, bounds, t_obs, init_cond, x_obs, y_obs):
    '''
    L-BFGS-B method to find optimized parameters by finding local minimum error (MAPE) based on initial conditions
    '''
    result = minimize(compute_mape, init_guess, args=(t_obs, init_cond, x_obs, y_obs), bounds=bounds, method='L-BFGS-B')
    return result.x

def global_optimization_RMSE(bounds, t_obs, init_cond, x_obs, y_obs):
    '''
    Simulated annealing to find optimized parameters by finding global minimum error (RMSE)
    '''
    result = dual_annealing(compute_combined_rmse, bounds, args=(t_obs, init_cond, x_obs, y_obs), seed=50)
    return result.x

def global_optimization_MAPE(bounds, t_obs, init_cond, x_obs, y_obs):
    '''
    Simulated annealing to find optimized parameters by finding global minimum error (MAPE)
    '''
    result = dual_annealing(compute_mape, bounds, args=(t_obs, init_cond, x_obs, y_obs), seed=50)
    return result.x