from src import load_data, simulate_model
from src.utils import plot_diffs

t_obs, x_obs, y_obs = load_data('data/predator-prey-data.csv')

init_cond=[x_obs[0], y_obs[0]]

print(init_cond)

init_guess = (0.1, 0.02, 0.3, 0.1)# Initial guesses for the model parameters 

sim_data = simulate_model(init_guess, init_cond, t_obs)

plot_diffs(t_obs,x_obs, y_obs, sim_data[0], sim_data[1]) 