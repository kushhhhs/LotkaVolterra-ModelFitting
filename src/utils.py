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

    plt.figure(figsize=(10, 10))

    plt.suptitle(f"Fitted Lotka-Volterra Model; Method: {method}", fontsize=16)

    # Plotting the differences between observed and simulated values for prey population
    plt.subplot(2, 1, 1)
    plt.plot(t_obs, x_obs, color=purple, label="Observed Prey Population", 
             linestyle='--', marker='o')       
    plt.plot(t_obs, x_simulated, color=yellow, label="Simulated Prey Population")
    plt.xlabel("Time")
    plt.ylabel("Prey Population")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plotting the differences between observed and simulated values for predator population
    plt.subplot(2, 1, 2)
    plt.plot(t_obs, y_obs, color=purple, label="Observed Predator Population", 
             linestyle='--', marker='o')
    plt.plot(t_obs, y_simulated, color=yellow, label="Simulated Predator Population")
    plt.xlabel("Time")
    plt.ylabel("Predator Population")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_sims(t, x, y):
    plt.figure(figsize=(10,6))

    plt.scatter(t, x, color='blue', label='Prey Population (x)', marker='o')
    plt.scatter(t, y, color='red', label='Predator Population (y)', marker='o')

    plt.xlabel(' Time ')
    plt.ylabel(' Population')
    plt.legend()
    plt.show()
    
def plot_diffs_subplot(ax1, ax2, method, t_obs, x_obs, y_obs, x_simulated, y_simulated):
    yellow = '#FFD700'
    purple = '#9370DB'

    # Plotting the differences between observed and simulated values for prey population
    ax1.plot(t_obs, x_obs, color=purple, label="Observed Prey Population", 
            linestyle='--', marker='o')       
    ax1.plot(t_obs, x_simulated, color=yellow, label="Simulated Prey Population")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Prey Population")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plotting the differences between observed and simulated values for predator population
    ax2.plot(t_obs, y_obs, color=purple, label="Observed Predator Population", 
            linestyle='--', marker='o')
    ax2.plot(t_obs, y_simulated, color=yellow, label="Simulated Predator Population")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Predator Population")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
