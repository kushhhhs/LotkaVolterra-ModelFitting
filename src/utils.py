import pandas as pd 
import matplotlib.pyplot as plt

def load_data(file_path):

    data = pd.read_csv(filepath_or_buffer= file_path)
    
    t_obs = data['t'].values
    x_obs = data['x'].values
    y_obs = data['y'].values

    return t_obs, x_obs, y_obs

def plot_diffs(t_obs, x_obs, y_obs, x_simulated, y_simulated):
    
    plt.figure(figsize=(10, 5))

    # Plotting the differences between observed and simulated values for prey population
    plt.subplot(1, 2, 1)
    plt.plot(t_obs, x_obs, label = "Observed Prey Population", linestyle='--', marker='o')       
    plt.plot(t_obs, x_simulated, label = "Simulated Prey Population")
    plt.xlabel("Time")
    plt.ylabel("Prey Population")
    plt.legend()

    # Plotting the differences between observed and simulated values for predator population
    plt.subplot(1, 2, 2)
    plt.plot(t_obs, y_obs, label="Observed Predator Population", linestyle='--', marker='o')
    plt.plot(t_obs, y_simulated, label="Simulated Predator Population")
    plt.xlabel("Time")
    plt.ylabel("Predator Population")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_sims(t, x, y):
    plt.figure(figsize=(10,6))

    plt.scatter(t, x, color='blue', label='Prey Population (x)', marker='o')
    plt.scatter(t, x, color='red', label='Predator Population (x)', marker='o')

    plt.xlabel(' Time ')
    plt.ylabel(' Population')
    plt.legend()
    plt.show()
