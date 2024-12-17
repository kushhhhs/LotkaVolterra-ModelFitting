'''
Lotka-Volterra Equations and Solver 
'''
from typing import List, Union
from scipy.integrate import odeint
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