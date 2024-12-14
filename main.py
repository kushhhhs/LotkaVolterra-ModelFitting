from src import load_data, simulate_model
from src.utils import plot_diffs
from src.optimization import local_optimization_RMSE, local_optimization_MAPE, global_optimization_RMSE, global_optimization_MAPE,compute_combined_rmse
import numpy as np 
t_obs, x_obs, y_obs = load_data('data/predator-prey-data.csv')


init_cond=[x_obs[0], y_obs[0]]

init_guess = (1.0, 0.5, 1.5, 1.0)
bounds = [(0, 2), (0, 2), (0, 2), (0, 2)]

optimized_params_local_RMSE = local_optimization_RMSE(init_guess, bounds, t_obs, init_cond, x_obs, y_obs)
optimized_params_global_RMSE = global_optimization_RMSE(bounds, t_obs, init_cond, x_obs, y_obs)
optimized_params_local_MAPE = local_optimization_MAPE(init_guess, bounds, t_obs, init_cond, x_obs, y_obs)
optimized_params_global_MAPE = global_optimization_MAPE(bounds, t_obs, init_cond, x_obs, y_obs)

sim_data_local_RMSE = simulate_model(optimized_params_local_RMSE, init_cond, t_obs)
sim_data_global_RMSE = simulate_model(optimized_params_global_RMSE, init_cond, t_obs)
sim_data_local_MAPE = simulate_model(optimized_params_local_MAPE, init_cond, t_obs)
sim_data_global_MAPE = simulate_model(optimized_params_global_MAPE, init_cond, t_obs)

plot_diffs(t_obs, x_obs, y_obs, sim_data_local_RMSE[0], sim_data_local_RMSE[1])
plot_diffs(t_obs, x_obs, y_obs, sim_data_global_RMSE[0], sim_data_global_RMSE[1])
plot_diffs(t_obs, x_obs, y_obs, sim_data_local_MAPE[0], sim_data_local_MAPE[1])
plot_diffs(t_obs, x_obs, y_obs, sim_data_global_MAPE[0], sim_data_global_MAPE[1])

def combined_removal(t_obs, x_obs, y_obs, init_cond, bounds, tol):
    removed_points = 0
    i = 0
    while i < len(t_obs) :
        # Removing the i-th time point and corresponding prey observation
        t_reduced = t_obs[i:]
        y_reduced = y_obs[i:]
        x_reduced = x_obs[i:]
        removed_points += 1
        # Fitting the parameters to the model using the reduced dataset
        print(len(t_reduced))
        optimized_params = global_optimization_RMSE(bounds, t_reduced, init_cond, x_reduced, y_reduced)
        
        combined_rmse = compute_combined_rmse(optimized_params, t_reduced, init_cond, x_reduced, y_reduced)
        
        if combined_rmse > tol:
            print(f"Stopping: RMSE exceeded threshold of {tol}.")
            break
        
        print(f"RMSE after removing {removed_points} time points = {combined_rmse:.4f}")
        i += 1
combined_removal(t_obs, x_obs, y_obs, init_cond, bounds, tol = 1)

