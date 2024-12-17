import numpy as np 
from scipy.interpolate import interp1d
def remove_random_points(t_obs, data, num_points):
    removed_indices = []
    for _ in range(num_points):    
        index_removed = np.random.randint(len(t_obs))
        removed_indices.append(index_removed)
        # Remove the corresponding point from both time series
        t_reduced = np.delete(t_obs, index_removed)
        data_reduced = np.delete(data, index_removed)
    
        print(f"Removed point at index {index_removed}: t = {t_obs[index_removed]:.2f}, x = {data[index_removed]:.5f}")
    
    return t_reduced, data_reduced, removed_indices

def interpolate_points(t_reduced, t_obs, reduced_data, data, index_removed):
    interpolator = interp1d(t_reduced, reduced_data, kind='cubic', fill_value="extrapolate")
    
    # Interpolate the removed point
    x_interpolated = interpolator(t_obs[index_removed])
    
    print(f"Interpolated value at t = {t_obs[index_removed]:.2f}: {x_interpolated:.5f}")
    print(type(x_interpolated))
    data[index_removed] = x_interpolated.item()
    return data