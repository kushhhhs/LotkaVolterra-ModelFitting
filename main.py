import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import matplotlib.pyplot as plt 
from src.utils import load_data, plot_diffs, plot_comparison
from src.model import simulate_model
from src.optimization import local_optimization,simulated_annealing

t_obs, x_obs, y_obs = load_data('data/predator-prey-data.csv')

init_cond=[x_obs[0], y_obs[0]]
init_guess = (0.1, 0.02, 0.3, 0.1)# Initial guesses for the model parameters
sim_data = simulate_model(params=init_guess, init_cond=init_cond, t_obs=t_obs)

plot_diffs(t_obs,x_obs, y_obs, sim_data[0], sim_data[1]) 

# Define parameter bounds (min, max) for each parameter
bounds = (
    (0.0, 1.0),  # alpha bounds
    (0.0, 1.0),  # beta bounds
    (0.0, 1.0),  # gamma bounds
    (0.0, 1.0)   # delta bounds
)

step_size = 0.01
max_iter = 1000 
tolerance = 1e-6

# Run local optimization
optimal_params, final_error, history = local_optimization(
    init_guess=init_guess,
    bounds=bounds,
    t_obs=t_obs,
    init_cond=init_cond,
    x_obs=x_obs,
    y_obs=y_obs,
    step_size=step_size,
    max_iter=max_iter,
    tolerance=tolerance
)

initial_sim_data = sim_data
optimized_sim_data = simulate_model(optimal_params, init_cond, t_obs)

# comparison plot
plot_comparison(
    t_obs, x_obs, y_obs,
    init_guess, optimal_params,
    initial_sim_data, optimized_sim_data,
    history, final_error
)

# Simulated Annealing hyperparameters
SA_TEMP_INIT = 1.0       # Initial temperature
SA_TEMP_FINAL = 0.01     # Final temperature
SA_ALPHA = 0.95          # Cooling rate
SA_MAX_ITER = 1000       # Maximum total iterations
SA_MAX_ITER_TEMP = 100   # Maximum iterations at each temperature
random_seed = 50

optimal_params_sa, final_error_sa, history_sa = simulated_annealing(
    init_guess=init_guess,
    bounds=bounds,
    t_obs=t_obs,
    init_cond=init_cond,
    x_obs=x_obs,
    y_obs=y_obs,
    temp_init = SA_TEMP_INIT,
    temp_final = SA_TEMP_FINAL,
    alpha = SA_ALPHA,
    max_iter = SA_MAX_ITER,
    max_iter_temp = SA_MAX_ITER_TEMP,
    random_seed = random_seed 
)
sa_optimized_sim_data = simulate_model(optimal_params_sa, init_cond, t_obs)

# Create comparison plot
plot_comparison(
    t_obs, x_obs, y_obs,
    init_guess, optimal_params_sa,
    initial_sim_data, sa_optimized_sim_data,
    history_sa, final_error_sa
)