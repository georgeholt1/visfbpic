# Author: George K. Holt
# License: MIT
# Version: 0.0.1
"""
Part of VISFBPIC.

Contains an example plasma density profile method.
"""
import numpy as np
from scipy.constants import pi

def number_density(z):
    '''For a given value of z-position, return number density n.'''
    n0 = 1.0e24
    ramp_length = 100e-6
    plateau_length = 8e-3
    
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