'''
Optimization Methods will have methods for parameter optimization 
'''
import numpy as np
from src.model import simulate_model
import time
from datetime import datetime
from typing import Tuple, List
import random

def compute_combined_rmse(
    params: tuple,
      t_obs: np.ndarray, 
      init_cond: list, 
      x_obs: np.ndarray,
      y_obs: np.ndarray)-> float:
    '''
    Compute error between observed and simulated population trajectories 

    Args:
    params (tuple): Model parameters (alpha, beta, gamma, delta)
    '''
    simulate = simulate_model(params, init_cond, t_obs)
    
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

def local_optimization(
    init_guess: tuple, 
    bounds: tuple,
    t_obs: np.ndarray, 
    init_cond: list, 
    x_obs: np.ndarray, 
    y_obs: np.ndarray,
    step_size: float,
    max_iter: int,
    tolerance: float 
) -> tuple:
    """
    Hill climbing local optimization for Lotka-Volterra parameters.
    
    Args:
        init_guess: Initial parameter values (alpha, beta, gamma, delta)
        bounds: Tuple of (min, max) bounds for parameters
        t_obs: Time points of observations
        init_cond: Initial conditions [x0, y0]
        x_obs: Observed prey population
        y_obs: Observed predator population
        step_size: Size of parameter adjustments
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        tuple: (best_params, best_error, convergence_history)
    """
      # Initialize timing
    start_time = time.time()
    last_update = start_time
    update_interval = 2  # Print update every 2 seconds

    # Initialize
    current_params = list(init_guess)
    current_error = compute_combined_rmse(current_params, t_obs, init_cond, x_obs, y_obs)
    
    best_params = current_params.copy()
    best_error = current_error
    
    # Track convergence
    convergence_history = [best_error]
    
    # Main optimization loop
    for iteration in range(max_iter):
        improved = False
        
        # Try to improve each parameter - Upper and Lower bound
        for i in range(len(current_params)):
            # Try increasing
            test_params = current_params.copy()
            test_params[i] += step_size
            
            # Check bounds
            if test_params[i] <= bounds[i][1]:  # Check upper bound
                test_error = compute_combined_rmse(
                    tuple(test_params), t_obs, init_cond, x_obs, y_obs
                )
                
                if test_error < current_error:
                    current_params = test_params
                    current_error = test_error
                    improved = True
                    continue
            
            # Try decreasing
            test_params = current_params.copy()
            test_params[i] -= step_size
            
            # Check bounds
            if test_params[i] >= bounds[i][0]:  # Check lower bound
                test_error = compute_combined_rmse(
                    tuple(test_params), t_obs, init_cond, x_obs, y_obs
                )
                
                if test_error < current_error:
                    current_params = test_params
                    current_error = test_error
                    improved = True
        
        # Update best solution if current is better
        if current_error < best_error:
            best_params = current_params.copy()
            best_error = current_error
        
        # Track progress
        convergence_history.append(best_error)
        
        # Termination conditions
        if not improved or best_error < tolerance:
            break

    # Final timing
    total_time = time.time() - start_time
    print(f"\nOptimization completed in {total_time:.1f} seconds")
    print(f"Final error: {best_error:.6f}")
    print(f"Number of iterations: {iteration + 1}")
    
    return tuple(best_params), best_error, convergence_history

def simulated_annealing(
    init_guess: tuple,
    bounds: tuple,
    t_obs: np.ndarray,
    init_cond: list,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    temp_init: float ,
    temp_final: float,
    alpha: float ,
    max_iter: int,
    max_iter_temp: int,
    random_seed :int,
) -> Tuple[tuple, float, List[float]]:
    """
    Global optimization using Simulated Annealing algorithm.
    
    Args:
        init_guess: Initial parameter values (alpha, beta, gamma, delta)
        bounds: Parameter bounds (min, max) for each parameter
        t_obs: Time points of observations
        init_cond: Initial conditions [x0, y0]
        x_obs: Observed prey population
        y_obs: Observed predator population
        temp_init: Initial temperature
        temp_final: Final temperature
        alpha: Cooling rate (0 < alpha < 1)
        max_iter: Maximum total iterations
        max_iter_temp: Maximum iterations at each temperature
    
    Returns:
        best_params: Optimized parameters
        best_error: Final error value
        history: List of error values during optimization
    """
    #reproduce
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Initialize
    current_params = list(init_guess)
    current_error = compute_combined_rmse(current_params, t_obs, init_cond, x_obs, y_obs)
    
    best_params = current_params.copy()
    best_error = current_error
    
    # Track optimization history
    history = [current_error]
    
    # Temperature schedule
    temp = temp_init
    iteration = 0
    
    # Main optimization loop
    while temp > temp_final and iteration < max_iter:
        # Iterations at current temperature
        for _ in range(max_iter_temp):
            # Generate neighbor solution by perturbing one random parameter
            neighbor_params = current_params.copy()
            param_idx = random.randint(0, len(current_params) - 1)
            
            # Perturbation size decreases with temperature
            perturbation = random.uniform(-0.1, 0.1) * temp
            neighbor_params[param_idx] += perturbation
            
            # Enforce bounds
            neighbor_params[param_idx] = max(bounds[param_idx][0], 
                                          min(bounds[param_idx][1], 
                                              neighbor_params[param_idx]))
            
            # Evaluate neighbor solution
            neighbor_error = compute_combined_rmse(neighbor_params, t_obs, init_cond, x_obs, y_obs)
            
            # Calculate acceptance probability
            delta_error = neighbor_error - current_error
            acceptance_prob = np.exp(-delta_error / temp) if temp > 0 else 0
            
            # Accept or reject neighbor solution
            if delta_error < 0 or random.random() < acceptance_prob:
                current_params = neighbor_params
                current_error = neighbor_error
                
                # Update best solution if improved
                if current_error < best_error:
                    best_params = current_params.copy()
                    best_error = current_error
            
            history.append(current_error)
            iteration += 1
            
            if iteration >= max_iter:
                break
        
        # Cool down
        temp *= alpha
        print(f"Temperature: {temp:.6f}, Best Error: {best_error:.6f}")
    
    return tuple(best_params), best_error, history