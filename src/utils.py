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
    
def plot_statistical_results(results):
    """
    Plot results with confidence intervals for each parameter
    
    Args:
        results: Dictionary containing:
            - 'prey_fixed': {'means': array, 'confs': array}
            - 'pred_fixed': {'means': array, 'confs': array}
            - 'num_deletions': list of deletion counts
    """
    param_names = ['α', 'β', 'γ', 'δ']
    num_deletions = results['num_deletions']
    
    plt.figure(figsize=(15, 10))
    for i, param in enumerate(param_names):
        plt.subplot(2, 2, i+1)
        
        # Plot prey fixed results (when we delete predator points)
        means_prey = results['prey_fixed']['means'][:, i]
        confs_prey = results['prey_fixed']['confs'][:, i]
        plt.fill_between(num_deletions, means_prey - confs_prey, means_prey + confs_prey, 
                        color='#FFD700', alpha=0.2, label='_nolegend_')
        plt.plot(num_deletions, means_prey, 'o-', color='#FFD700', 
                label='Deleting Predator Points')
        
        # Plot predator fixed results (when we delete prey points)
        means_pred = results['pred_fixed']['means'][:, i]
        confs_pred = results['pred_fixed']['confs'][:, i]
        plt.fill_between(num_deletions, means_pred - confs_pred, means_pred + confs_pred, 
                        color='#9370DB', alpha=0.2, label='_nolegend_')
        plt.plot(num_deletions, means_pred, 'o-', color='#9370DB', 
                label='Deleting Prey Points')
        
        plt.xlabel('Points Removed')
        plt.ylabel(f'Parameter {param}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add title for each parameter
        plt.title(f'Parameter {param} vs Points Removed')
    
    plt.suptitle('Parameter Sensitivity to Data Point Removal', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_parameter_comparison(results, baseline_params, baseline_conf):
    """
    Plot parameter changes with baseline comparison
    """
    param_names = ['α', 'β', 'γ', 'δ']
    num_deletions = results['num_deletions']
    
    plt.figure(figsize=(15, 10))
    plt.suptitle('Optimized Parameters for predator data deletion', fontsize=14)
    
    for i, param in enumerate(param_names):
        plt.subplot(2, 2, i+1)
        
        # Plot baseline
        plt.axhline(y=baseline_params[i], color='blue', label=f'{param} baseline')
        plt.fill_between([0, 100], 
                        baseline_params[i] - baseline_conf[i],
                        baseline_params[i] + baseline_conf[i],
                        color='lightblue', alpha=0.3)
        
        # Plot deletion results
        means = results['prey_fixed']['means'][:, i]
        confs = results['prey_fixed']['confs'][:, i]
        plt.errorbar(num_deletions, means, yerr=confs, 
                    fmt='ro', label=f'{param} with deletion',
                    capsize=3, capthick=1, elinewidth=1)
        
        plt.xlabel('Number of deleted points')
        plt.ylabel(f'Parameter {param}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
    plt.tight_layout()
    plt.show()