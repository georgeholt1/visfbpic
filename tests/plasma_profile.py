import numpy as np
from scipy.constants import pi

def number_density(z):
    n0 = 1e24
    ramp_length = 100e-6
    plateau_length = 2e-3
    
    if z < 0:
        n = 0
    elif z < ramp_length:
        n =  0.5 * (1 - np.cos(pi * z / ramp_length))
    elif z < ramp_length + plateau_length:
        n = 1
    elif z < 2 * ramp_length + plateau_length:
        n = 0.5 * (1 + np.cos(pi * (z - (ramp_length + plateau_length)) / \
            ramp_length))
    else:
        n = 0
    n *= n0
    return n
