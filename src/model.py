'''
Lotka-Volterra Equations and Solver 
'''
from typing import List, Union
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde 

def lotka_volterra(state, t, alpha: float, beta: float, gamma: float, delta: float):
    '''
    Function to compute the derivatives od the Lotka-Volterra model 
    Args:
        t(float): Time point (is required for most numerical integration methods)
        state(List[float]): Current population state containing the (predator poulation, prey population)
        alpha(float): Prey population growth rate
        beta(float): Prey population death rate
        gamma(float): Predator population death rate
        delta(float): Predator population growth rate 
    Returns :
        The change in prey and predator population
    '''
    x, y = state
    dx_dt = alpha * x - beta * x * y
    dy_dt = delta * x * y - gamma * y
    return [dx_dt, dy_dt]
 
def simulate_model(params: tuple, init_cond, t_obs):
    '''
    Solve the Lotka-Volterra equations with params using a numerical integrator (return simulated x and y populations)
    '''
    alpha, beta, gamma, delta = params
    sol = odeint(lotka_volterra, init_cond, t_obs, args=(alpha, beta, gamma, delta))
    return sol[:, 0], sol[:, 1]


def objective_function(params, t_data, x_data, y_data):
    """Objective function to minimize"""
    alpha, beta, gamma, delta = params
    x0, y0 = x_data[0], y_data[0]
    
    # Simulate model
    model_result = odeint(lotka_volterra, [x0, y0], t_data, args=(alpha, beta, gamma, delta))
    x_model, y_model = model_result.T
    
    # Calculate sum of squared errors
    sse_x = np.sum((x_model - x_data) ** 2)
    sse_y = np.sum((y_model - y_data) ** 2)
    
    # Calculate KDE difference
    x_kde = gaussian_kde(x_data)
    y_kde = gaussian_kde(y_data)
    x_model_kde = gaussian_kde(x_model)
    y_model_kde = gaussian_kde(y_model)
    kde_diff = np.sum((x_kde(x_data) - x_model_kde(x_data)) ** 2) + \
               np.sum((y_kde(y_data) - y_model_kde(y_data)) ** 2)
    
    return sse_x + sse_y + kde_diff

def optimize_lotka_volterra(t_data, x_data, y_data, init_params):
    """Optimize Lotka-Volterra model parameters"""
    res = minimize(objective_function, init_params, args=(t_data, x_data, y_data),
                   method='L-BFGS-B', bounds=((0, None), (0, None), (0, None), (0, None)))
    return res.x

def sensitivity_analysis(t_data, x_data, y_data, params):
    """Perform sensitivity analysis"""
    alpha, beta, gamma, delta = params
    
    # Compute model predictions
    x0, y0 = x_data[0], y_data[0]
    model_result = odeint(lotka_volterra, [x0, y0], t_data, args=(alpha, beta, gamma, delta))
    x_model, y_model = model_result.T
    
    # Compute parameter sensitivities
    sensitivities = {}
    for param, value in zip(['alpha', 'beta', 'gamma', 'delta'], [alpha, beta, gamma, delta]):
        perturbed_params = [alpha, beta, gamma, delta]
        perturbed_params[['alpha', 'beta', 'gamma', 'delta'].index(param)] += 0.01 * value
        perturbed_result = odeint(lotka_volterra, [x0, y0], t_data, args=tuple(perturbed_params))
        perturbed_x, perturbed_y = perturbed_result.T
        sensitivities[param] = {
            'x': (perturbed_x - x_model) / (0.01 * value),
            'y': (perturbed_y - y_model) / (0.01 * value)
        }
    
    # Identify critical data points
    data_point_sensitivities = []
    for i in range(len(t_data)):
        mask = np.ones(len(t_data), dtype=bool)
        mask[i] = False
        reduced_params = optimize_lotka_volterra(t_data[mask], x_data[mask], y_data[mask], params)
        param_diff = np.linalg.norm(reduced_params - params)
        data_point_sensitivities.append(param_diff)
    
    return sensitivities, np.array(data_point_sensitivities)

def objective_function(params, t_data, x_data, y_data):
    """Objective function to minimize"""
    alpha, beta, gamma, delta = params
    x0, y0 = x_data[0], y_data[0]
    
    # Simulate model
    model_result = odeint(lotka_volterra, [x0, y0], t_data, args=(alpha, beta, gamma, delta))
    x_model, y_model = model_result.T
    
    # Calculate sum of squared errors
    sse_x = np.sum((x_model - x_data) ** 2)
    sse_y = np.sum((y_model - y_data) ** 2)
    
    # Calculate KDE difference
    x_kde = gaussian_kde(x_data)
    y_kde = gaussian_kde(y_data)
    x_model_kde = gaussian_kde(x_model)
    y_model_kde = gaussian_kde(y_model)
    kde_diff = np.sum((x_kde(x_data) - x_model_kde(x_data)) ** 2) + \
               np.sum((y_kde(y_data) - y_model_kde(y_data)) ** 2)
    
    return sse_x + sse_y + kde_diff

def optimize_lotka_volterra(t_data, x_data, y_data, init_params):
    """Optimize Lotka-Volterra model parameters"""
    res = minimize(objective_function, init_params, args=(t_data, x_data, y_data),
                   method='L-BFGS-B', bounds=((0, None), (0, None), (0, None), (0, None)))
    return res.x

def sensitivity_analysis(t_data, x_data, y_data, params):
    """Perform sensitivity analysis"""
    alpha, beta, gamma, delta = params
    
    # Compute model predictions
    x0, y0 = x_data[0], y_data[0]
    model_result = odeint(lotka_volterra, [x0, y0], t_data, args=(alpha, beta, gamma, delta))
    x_model, y_model = model_result.T
    
    # Compute parameter sensitivities
    sensitivities = {}
    for param, value in zip(['alpha', 'beta', 'gamma', 'delta'], [alpha, beta, gamma, delta]):
        perturbed_params = [alpha, beta, gamma, delta]
        perturbed_params[['alpha', 'beta', 'gamma', 'delta'].index(param)] += 0.01 * value
        perturbed_result = odeint(lotka_volterra, [x0, y0], t_data, args=tuple(perturbed_params))
        perturbed_x, perturbed_y = perturbed_result.T
        sensitivities[param] = {
            'x': (perturbed_x - x_model) / (0.01 * value),
            'y': (perturbed_y - y_model) / (0.01 * value)
        }
    
    # Identify critical data points
    data_point_sensitivities = []
    for i in range(len(t_data)):
        mask = np.ones(len(t_data), dtype=bool)
        mask[i] = False
        reduced_params = optimize_lotka_volterra(t_data[mask], x_data[mask], y_data[mask], params)
        param_diff = np.linalg.norm(reduced_params - params)
        data_point_sensitivities.append(param_diff)
    
    return sensitivities, np.array(data_point_sensitivities)
