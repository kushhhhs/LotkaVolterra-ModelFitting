'''
Optimization Methods will have methods for parameter optimization 
'''
import numpy as np
from src.model import simulate_model
from scipy.optimize import minimize, dual_annealing
from joblib import Parallel, delayed
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import t
from typing import Dict, List, Tuple
from datetime import datetime



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
#-------------------------------------------------------------------------------------------------------------------- Testing Optimization Functions 
def calc_mean_conf(params: np.ndarray, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and confidence intervals for parameters
    
    Args:
        params: Array of shape (n_trials, n_parameters) containing parameter estimates
        confidence_level: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        means: Array of mean parameter values
        confs: Array of confidence interval half-widths
    """
    n_trials = len(params)
    means = np.mean(params, axis=0)
    
    # Calculate standard error
    std_err = np.std(params, axis=0, ddof=1) / np.sqrt(n_trials)
    
    # Get t-value for desired confidence level
    t_value = t.ppf((1 + confidence_level) / 2, n_trials - 1)
    
    # Calculate confidence intervals
    confs = t_value * std_err
    
    return means, confs


def compute_combined_rmse_with_deletion(params: tuple, t_obs: np.ndarray, init_cond: list, 
                                      x_obs: np.ndarray, y_obs: np.ndarray) -> float:
    """
    Compute RMSE between observed and simulated populations
    All arrays must have the same length
    """
    x_simulated, y_simulated = simulate_model(params, init_cond, t_obs)
    
    # Compute errors
    x_error = np.mean((x_obs - x_simulated) ** 2)
    y_error = np.mean((y_obs - y_simulated) ** 2)
    
    return x_error + y_error

def run_single_trial(seed: int, t_obs: np.ndarray, x_obs: np.ndarray, 
                    y_obs: np.ndarray, init_cond: List[float],
                    bounds: List[Tuple[float, float]]) -> np.ndarray:
    """Run single optimization trial"""
    result = dual_annealing(
        compute_combined_rmse_with_deletion,
        bounds,
        args=(t_obs, init_cond, x_obs, y_obs),
        seed=seed
    )
    return result.x

def optimization_with_deletion(t_data: np.ndarray, X_data: np.ndarray, Y_data: np.ndarray,
                             init_cond: List[float], num_deletions: List[int], 
                             bounds: List[Tuple[float, float]], fix_prey: bool = True,
                             n_trials: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate optimal parameters with proper seed differentiation for prey/predator deletion
    """
    mean_params = np.zeros((len(num_deletions), 4))
    conf_params = np.zeros((len(num_deletions), 4))
    
    # Create DataFrame to store results
    results_df = pd.DataFrame(columns=[
        'points_removed', 
        'alpha_mean', 'alpha_conf',
        'beta_mean', 'beta_conf',
        'gamma_mean', 'gamma_conf',
        'delta_mean', 'delta_conf'
    ])
    
    population_type = "predator" if fix_prey else "prey"
    print(f"\nStarting {population_type} deletion analysis at {datetime.now().strftime('%H:%M:%S')}")

    for i, deletions in enumerate(num_deletions):
        print(f"\nScenario {i+1}/{len(num_deletions)}: Removing {deletions} points...")
        
        # Different seed bases for predator vs prey deletion
        base_seed = i * 100 if fix_prey else i * 100 + 1000
        np.random.seed(base_seed)
        
        # Generate deletion indices (exclude initial point)
        delete_indices = np.random.choice(range(1, len(Y_data)), size=deletions, replace=False)
        mask = np.ones(len(t_data), dtype=bool)
        mask[delete_indices] = False
        
        # Create reduced datasets
        t_reduced = t_data[mask]
        X_reduced = X_data[mask]
        Y_reduced = Y_data[mask]
            
        # Run parallel optimizations with different seeds
        deletion_params = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_single_trial)(
                base_seed + seed,  # Different seed for each trial
                t_reduced,
                X_reduced,
                Y_reduced,
                init_cond,
                bounds
            ) for seed in range(n_trials)
        )

        # Calculate statistics
        deletion_params = np.array(deletion_params)
        means, confs = calc_mean_conf(deletion_params)
        
        mean_params[i, :] = means
        conf_params[i, :] = confs
        
        # Add results to DataFrame
        results_df.loc[i] = [
            deletions,
            means[0], confs[0],  # alpha
            means[1], confs[1],  # beta
            means[2], confs[2],  # gamma
            means[3], confs[3]   # delta
        ]
        
        print(f"\nTrial results for {deletions} points removed:")
        print("Parameter values across trials:")
        param_names = ['α', 'β', 'γ', 'δ']
        for j, param in enumerate(param_names):
            print(f"{param} values:", deletion_params[:, j])
        
        print(f"\nOptimized parameters:")
        print(f"- α: {means[0]:.4f} ± {confs[0]:.4f}")
        print(f"- β: {means[1]:.4f} ± {confs[1]:.4f}")
        print(f"- γ: {means[2]:.4f} ± {confs[2]:.4f}")
        print(f"- δ: {means[3]:.4f} ± {confs[3]:.4f}")

    print(f"\nAnalysis complete for {population_type} deletion!")
    print("\nResults Summary:")
    print(results_df.to_string(index=False))
    
    return mean_params, conf_params, results_df

def optimization_with_combined_deletion(t_data: np.ndarray, X_data: np.ndarray, Y_data: np.ndarray,
                                      init_cond: List[float], num_deletions: List[int], 
                                      bounds: List[Tuple[float, float]], n_trials: int = 10):
    """
    Calculate optimal parameters when deleting points from both time series
    """
    mean_params = np.zeros((len(num_deletions), 4))
    conf_params = np.zeros((len(num_deletions), 4))
    
    print("\nStarting combined deletion analysis...")
    
    for i, deletions in enumerate(num_deletions):
        print(f"\nScenario {i+1}/{len(num_deletions)}: Removing {deletions} points from each time series...")
        
        # Generate different deletion indices for each time series
        np.random.seed(i * 100)  # Seed for prey deletions
        prey_delete_indices = np.random.choice(range(1, len(X_data)), size=deletions, replace=False)
        
        np.random.seed(i * 100 + 50)  # Different seed for predator deletions
        pred_delete_indices = np.random.choice(range(1, len(Y_data)), size=deletions, replace=False)
        
        # Create masks
        prey_mask = np.ones(len(t_data), dtype=bool)
        pred_mask = np.ones(len(t_data), dtype=bool)
        prey_mask[prey_delete_indices] = False
        pred_mask[pred_delete_indices] = False
        
        # Combined mask (keep points present in both series)
        combined_mask = prey_mask & pred_mask
        
        # Create reduced datasets
        t_reduced = t_data[combined_mask]
        X_reduced = X_data[combined_mask]
        Y_reduced = Y_data[combined_mask]
        
        print(f"Points remaining: {np.sum(combined_mask)}")
            
        # Run parallel optimizations
        deletion_params = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_single_trial)(
                (i * 100) + seed,
                t_reduced,
                X_reduced,
                Y_reduced,
                init_cond,
                bounds
            ) for seed in range(n_trials)
        )
        
        # Calculate statistics
        deletion_params = np.array(deletion_params)
        means, confs = calc_mean_conf(deletion_params)
        
        mean_params[i, :] = means
        conf_params[i, :] = confs
        
        print(f"\nOptimized parameters:")
        param_names = ['α', 'β', 'γ', 'δ']
        for j, param in enumerate(param_names):
            print(f"- {param}: {means[j]:.4f} ± {confs[j]:.4f}")
    
    return mean_params, conf_params

