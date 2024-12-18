import pandas as pd 
import matplotlib.pyplot as plt

def load_data(file_path):

    data = pd.read_csv(filepath_or_buffer= file_path)
    
    t_obs = data['t'].values
    x_obs = data['x'].values
    y_obs = data['y'].values

    return t_obs, x_obs, y_obs

def plot_diffs(method, t_obs, x_obs, y_obs, x_simulated, y_simulated):

    yellow = '#FFD700'
    purple = '#9370DB'

    # Calculate R-squared for both predator and prey
    ss_res_x = np.sum((x_obs - x_simulated) ** 2)
    ss_tot_x = np.sum((x_obs - np.mean(x_obs)) ** 2)
    r2_x = 1 - ss_res_x / ss_tot_x

    ss_res_y = np.sum((y_obs - y_simulated) ** 2)
    ss_tot_y = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2_y = 1 - ss_res_y / ss_tot_y

    # Create the plot
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"Fitted Lotka-Volterra Model; Method: {method}", fontsize=16)

    # Prey Population Plot
    plt.subplot(2, 1, 1)
    plt.plot(t_obs, x_obs, color=purple, label="Observed Prey Population", linestyle='--', marker='o')
    plt.plot(t_obs, x_simulated, color=yellow, label="Simulated Prey Population")
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Prey Population", fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"Prey Population (R² = {r2_x:.2f})", fontsize=16)

    # Predator Population Plot
    plt.subplot(2, 1, 2)
    plt.plot(t_obs, y_obs, color=purple, label="Observed Predator Population", linestyle='--', marker='o')
    plt.plot(t_obs, y_simulated, color=yellow, label="Simulated Predator Population")
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Predator Population", fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"Predator Population (R² = {r2_y:.2f})", fontsize=16)

    plt.tight_layout()
    plt.show()
