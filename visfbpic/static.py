# Author: George K. Holt
# License: MIT
# Version: 0.1.0
"""
Part of VISFBPIC.

Contains methods for creating static visualisations of the FBPIC data.
"""
import os
import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(os.path.join(
    os.path.split(__file__)[0], '_mpl_config', 'style.mplstyle'
))



def plot_plasma_profile(
    n_file,
    z_min,
    z_max,
    z_points=1000,
    out_dir=None,
    dpi=200,
    figsize=(8, 4.5)
):
    '''Create plot of plasma density profile.
    
    Parameters
    n_file : str
        Path to file containing a function called `number_density`. The function
        should take one argument: a value of z-position, and should return a
        corresponding number density value. See `plasma_profile_example.py` for
        an example.
    z_min, z_max : float
        Minimum and maximum values of the z-coordinate for the plasma profile.
    z_points : int, optional
        Number of points to sample the number density profile along z.
    out_dir : str, optional
        Path to output directory within which to save the animation. Defaults to
        None, which shows the visualisation onscreen.
    dpi : int, optional
        Dots per inch resolution. Changing this parameter may result in a bad
        plot layout. Defaults to 200.
    figsize : tuple, optional
        Figure size in the form (width, height). Changing this parameter may
        results in a bad plot layout. Defaults to (8, 4.5).
    '''
    # initialise method from external file
    spec = importlib.util.spec_from_file_location(
        "plasma_profile",
        n_file
    )
    number_density_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(number_density_mod)
    
    # generate data
    z_array = np.linspace(z_min, z_max, z_points)
    n_array = []
    for z in z_array:
        n_array.append(number_density_mod.number_density(z))
    n_array = np.array(n_array)
    n_profile_max_oom = int(np.floor(np.log10(n_array.max())))
    
    # plotting
    fig, ax = plt.subplots()
    ax.plot(
        z_array*1e3,
        n_array/10**n_profile_max_oom,
        c='C1'
    )
    
    ax.set_xlabel('$z$ (mm)')
    ax.set_ylabel(r'$n_e$ ($\times 10 ^ {{{}}}$'.format(
        n_profile_max_oom-6
    ) + ' cm$^{-3}$)')
    
    if out_dir is None:
        plt.show()
    else:
        fig.savefig(os.path.join(out_dir, "plasma_profile.png"))
        
        
        
# unit testing
if __name__ == "__main__":
    plot_plasma_profile(
        sys.argv[1],
        float(sys.argv[2]),
        float(sys.argv[3]),
        out_dir=sys.argv[4],
    )