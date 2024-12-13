from src import load_data, simulate_model
from src.utils import plot_diffs
from src.optimization import local_optimization_RMSE, local_optimization_MAPE, global_optimization_RMSE, global_optimization_MAPE

t_obs, x_obs, y_obs = load_data('data/predator-prey-data.csv')

init_cond=[x_obs[0], y_obs[0]]

init_guess = (0.1, 0.02, 0.3, 0.1)
bounds = [(0, 2), (0, 1), (0, 2), (0, 1)]

optimized_params_global_RMSE = global_optimization_RMSE(bounds, t_obs, init_cond, x_obs, y_obs)

sim_data = simulate_model(global_optimization_RMSE, init_cond, t_obs)

plot_diffs(t_obs, x_obs, y_obs, sim_data[0], sim_data[1])
