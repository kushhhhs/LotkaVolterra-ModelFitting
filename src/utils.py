import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def load_data(file_path):

    data = pd.read_csv(filepath_or_buffer= file_path)
    
    t_obs = data['t'].values
    x_obs = data['x'].values
    y_obs = data['y'].values

    return t_obs, x_obs, y_obs

def plot_diffs(method, t_obs, x_obs, y_obs, x_simulated, y_simulated):
    
    yellow = '#FFD700'
    purple = '#9370DB'

    # Detailed R-squared calculation with print statements
    print(f"\nDetailed R² calculation for {method}:")
    
    # Prey R² calculation
    ss_res_x = np.sum((x_obs - x_simulated) ** 2)
    ss_tot_x = np.sum((x_obs - np.mean(x_obs)) ** 2)
    r2_x = 1 - ss_res_x / ss_tot_x
    
    print("\nPrey:")
    print(f"Sum of squared residuals (ss_res_x): {ss_res_x}")
    print(f"Total sum of squares (ss_tot_x): {ss_tot_x}")
    print(f"Mean of observed prey: {np.mean(x_obs)}")
    print(f"R² calculation: 1 - {ss_res_x} / {ss_tot_x} = {r2_x}")

    # Predator R² calculation
    ss_res_y = np.sum((y_obs - y_simulated) ** 2)
    ss_tot_y = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2_y = 1 - ss_res_y / ss_tot_y
    
    print("\nPredator:")
    print(f"Sum of squared residuals (ss_res_y): {ss_res_y}")
    print(f"Total sum of squares (ss_tot_y): {ss_tot_y}")
    print(f"Mean of observed predator: {np.mean(y_obs)}")
    print(f"R² calculation: 1 - {ss_res_y} / {ss_tot_y} = {r2_y}")
    
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
    
    
def plot_deletion_results(num_deletions, mean_params, conf_params, baseline_params, baseline_conf, deletion_type='predator'):
    """
    Plot parameter changes with non-overlapping x-axis labels
    """
    yellow = '#FFD700'
    purple = '#9370DB'
    
    param_names = ['α', 'β', 'γ', 'δ']
    titles = ['Parameter ' + param for param in param_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Optimized Parameters for {deletion_type} data deletion', fontsize=12)
    axes = axes.ravel()
    
    print("Baseline parameters:")
    for param, val, conf in zip(param_names, baseline_params, baseline_conf):
        print(f"{param}: {val:.4f} ± {conf:.4f}")
    
    for i, (param, title) in enumerate(zip(param_names, titles)):
        ax = axes[i]
        
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Plot baseline
        ax.axhline(y=baseline_params[i], color=yellow, label=f'{param} baseline', zorder=1, linewidth=2)
        ax.fill_between([-10, 110], 
                       baseline_params[i] - baseline_conf[i],
                       baseline_params[i] + baseline_conf[i],
                       color=yellow, alpha=0.2, zorder=0,
                       label='Baseline 95% CI')
        
        # Stability threshold
        ax.axvline(x=60, color='gray', linestyle='--', alpha=0.5, 
                  label='Stability threshold', zorder=0)
        
        # Plot deletion results
        ax.errorbar(num_deletions, mean_params[:, i], yerr=conf_params[:, i],
                   fmt='o', color=purple, label=f'{param} with deletion',
                   capsize=3, capthick=1, elinewidth=1, zorder=2,
                   markersize=8, markeredgewidth=1.5)
        
        ax.set_xlabel('Number of deleted points', fontsize=10)
        ax.set_ylabel(f'Parameter {param}', fontsize=10)
        
        ax.legend(loc='best', framealpha=0.9, fontsize=9)
        
        # Y-axis limits
        ymin = min(baseline_params[i] - baseline_conf[i], 
                  np.min(mean_params[:, i] - conf_params[:, i]))
        ymax = max(baseline_params[i] + baseline_conf[i], 
                  np.max(mean_params[:, i] + conf_params[:, i]))
        padding = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - padding, ymax + padding)
        
        # X-axis setup with non-overlapping labels
        ax.set_xlim(-10, 110)
        ax.set_xticks([0, 20, 40, 60, 80, 85, 88, 90, 95])
        
        # Rotate x-axis labels slightly to prevent overlap
        ax.tick_params(axis='x', rotation=75)
        
        ax.set_title(title, pad=10, fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
def visualize_predator_prey_data(t_data, x_data, y_data):
    """
    Visualize the predator-prey data using KDE and 3D density plots in a single figure.
    
    Parameters:
    - t_data: Time series data
    - x_data: Predator population data
    - y_data: Prey population data
    """
    # KDE Density Plot
    fig = plt.figure(figsize=(16, 8))
    
    # 2D KDE Density Plot
    ax1 = fig.add_subplot(1, 2, 1)
    kde = gaussian_kde([x_data, y_data])
    x, y = np.mgrid[x_data.min():x_data.max():100j, y_data.min():y_data.max():100j]
    positions = np.vstack([x.ravel(), y.ravel()])
    z = kde(positions).reshape(x.shape)
    ax1.contourf(x, y, z, levels=20, cmap='viridis')
    ax1.scatter(x_data, y_data, c='red', alpha=0.5, label='Observed Data')
    ax1.set_xlabel('Predator Population')
    ax1.set_ylabel('Prey Population')
    ax1.set_title('KDE Density Plot')
    ax1.legend()
    
    # 3D Density Surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis')
    ax2.set_xlabel('Predator Population')
    ax2.set_ylabel('Prey Population')
    ax2.set_zlabel('Density')
    ax2.set_title('3D Density Surface')
    
    plt.tight_layout()
    plt.show()

def identify_critical_data_points(data_point_sensitivities, threshold=None):
    """
    Identify the critical data points based on the data point sensitivities.
    
    Parameters:
    - data_point_sensitivities (np.array): Array of data point sensitivities
    - threshold (float, optional): Sensitivity threshold to determine critical points.
                                  If not provided, the median sensitivity is used.
    
    Returns:
    - np.array: Array of indices of the critical data points
    """
    if threshold is None:
        threshold = np.median(data_point_sensitivities)

    critical_indices = np.where(data_point_sensitivities > threshold)[0]
    return critical_indices
