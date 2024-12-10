import pandas as pd 
import matplotlib.pyplot as plt

def load_data(file_path):

    data = pd.read_csv(filepath_or_buffer= file_path)
    
    t_obs = data['t'].values
    x_obs = data['x'].values
    y_obs = data['y'].values

    return t_obs, x_obs, y_obs

def plot_diffs(t_obs, x_obs, y_obs, x_simulated, y_simulated):
    """Plot the differences between observed and simulated populations."""
    prey_color = '#FFD700'     # Golden yellow
    predator_color = '#9370DB'  # Medium purple
    
    plt.figure(figsize=(10, 10))

    # Plotting the differences between observed and simulated values for prey population
    plt.subplot(2, 1, 1)
    plt.plot(t_obs, x_obs, color=prey_color, label="Observed Prey Population", 
             linestyle='--', marker='o')       
    plt.plot(t_obs, x_simulated, color=prey_color, label="Simulated Prey Population")
    plt.xlabel("Time")
    plt.ylabel("Prey Population")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plotting the differences between observed and simulated values for predator population
    plt.subplot(2, 1, 2)
    plt.plot(t_obs, y_obs, color=predator_color, label="Observed Predator Population", 
             linestyle='--', marker='o')
    plt.plot(t_obs, y_simulated, color=predator_color, label="Simulated Predator Population")
    plt.xlabel("Time")
    plt.ylabel("Predator Population")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_sims(t, x, y):
    """Plot simulated populations."""

    prey_color = '#FFD700'     # Golden yellow
    predator_color = '#9370DB'  # Medium purple
    
    plt.figure(figsize=(10,6))

    plt.scatter(t, x, color=prey_color, label='Prey Population', marker='o')
    plt.scatter(t, y, color=predator_color, label='Predator Population', marker='o')  

    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_comparison(t_obs, x_obs, y_obs, init_guess, optimal_params, initial_sim_data, optimized_sim_data, history, final_error):
    """
    Create side-by-side comparison plots of initial and optimized model fits.
    """

    bright_yellow = '#FFD700'  # Golden yellow
    bright_purple = '#9370DB'  # Medium purple
    dark_yellow = '#DAA520'    # Darker golden yellow for lines
    dark_purple = '#663399'    # Darker purple for lines

    # Create a figure with two subplots side by side
    plt.figure(figsize=(15, 6))

    # Plot original fit
    plt.subplot(1, 2, 1)
    plt.plot(t_obs, x_obs, color=bright_yellow, linestyle='--', marker='o', label='Observed Prey', alpha=1)
    plt.plot(t_obs, y_obs, color=bright_purple, linestyle='--', marker='o', label='Observed Predator', alpha=1)
    plt.plot(t_obs, initial_sim_data[0], color=dark_yellow, label='Initial Prey Fit', linewidth=2)
    plt.plot(t_obs, initial_sim_data[1], color=dark_purple, label='Initial Predator Fit', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(f'Initial Fit (Error: {history[0]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot optimized fit
    plt.subplot(1, 2, 2)
    plt.plot(t_obs, x_obs, color=bright_yellow, linestyle='--', marker='o', label='Observed Prey', alpha=1)
    plt.plot(t_obs, y_obs, color=bright_purple, linestyle='--', marker='o', label='Observed Predator', alpha=1)
    plt.plot(t_obs, optimized_sim_data[0], color=dark_yellow, label='Optimized Prey Fit', linewidth=2)
    plt.plot(t_obs, optimized_sim_data[1], color=dark_purple, label='Optimized Predator Fit', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(f'Optimized Fit (Error: {final_error:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print parameters
    print("\nParameters comparison:")
    print(f"Initial parameters: {init_guess}")
    print(f"Optimized parameters: {optimal_params}")