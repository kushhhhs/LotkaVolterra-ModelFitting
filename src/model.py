'''
Lotka-Volterra Equations and Solver 
'''
from typing import List, Union
from scipy.integrate import solve_ivp

def lotka_volterra(t: float, state: List[float], alpha: float, beta: float, gamma: float, delta: float)-> List[float]:
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
    Solve the Lotka-Volterra equations with params using a numerical integrator
    
    Args:
        params: Tuple of (alpha, beta, gamma, delta)
        init_cond: Initial conditions [x0, y0]
        t_obs: Time points for simulation
        
    Returns:
        Array of simulated x and y populations
    '''
    alpha, beta, gamma, delta = params
    sol = solve_ivp(lotka_volterra, (t_obs[0], t_obs[-1]), init_cond, args = (alpha, beta, gamma, delta), t_eval= t_obs)
    return sol.y
