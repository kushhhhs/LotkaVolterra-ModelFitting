from src import load_data, simulate_model
from src.utils import plot_diffs
from src.optimization import local_optimization_RMSE, local_optimization_MAPE, global_optimization_RMSE, global_optimization_MAPE

init_cond = [x_obs[0], y_obs[0]]

#init_guess = (1.0, 0.5, 1.5, 1.0) #This guess was found using global optimization methtod to improve Nelder-Mead method. 
init_guess = (1.7, 0.8, 1.4, 0.6) # This is a 'best initial guess' using previous biology/predator-prey knowledge.
bounds = [(0, 2), (0, 2), (0, 2), (0, 2)]

optimized_params_local_RMSE = local_optimization_RMSE(init_guess, bounds, t_obs, init_cond, x_obs, y_obs)
optimized_params_global_RMSE = global_optimization_RMSE(bounds, t_obs, init_cond, x_obs, y_obs)
optimized_params_local_MAPE = local_optimization_MAPE(init_guess, bounds, t_obs, init_cond, x_obs, y_obs)
optimized_params_global_MAPE = global_optimization_MAPE(bounds, t_obs, init_cond, x_obs, y_obs)

sim_data_local_RMSE = simulate_model(optimized_params_local_RMSE, init_cond, t_obs)
sim_data_global_RMSE = simulate_model(optimized_params_global_RMSE, init_cond, t_obs)
sim_data_local_MAPE = simulate_model(optimized_params_local_MAPE, init_cond, t_obs)
sim_data_global_MAPE = simulate_model(optimized_params_global_MAPE, init_cond, t_obs)

method = "Nelder-Mead (RMSE)"
plot_diffs(method, t_obs, x_obs, y_obs, sim_data_local_RMSE[0], sim_data_local_RMSE[1])
print(optimized_params_local_RMSE)
method = "Simulated Annealing (RMSE)"
plot_diffs(method, t_obs, x_obs, y_obs, sim_data_global_RMSE[0], sim_data_global_RMSE[1])
print(optimized_params_global_RMSE)
method = "Nelder-Mead (MAPE)"
plot_diffs(method, t_obs, x_obs, y_obs, sim_data_local_MAPE[0], sim_data_local_MAPE[1])
print(optimized_params_local_MAPE)
method = "Simulated Annealing (MAPE)"
plot_diffs(method, t_obs, x_obs, y_obs, sim_data_global_MAPE[0], sim_data_global_MAPE[1])
print(optimized_params_global_MAPE) 