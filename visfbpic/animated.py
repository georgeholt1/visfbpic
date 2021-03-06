# Author: George K. Holt
# License: MIT
# Version: 0.1.0
"""
Part of VISFBPIC.

Contains methods for creating animations of the FBPIC data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import animation
from matplotlib.ticker import FormatStrFormatter
from scipy.constants import e, c, pi
from tqdm import tqdm
import sys
import importlib.util
import csv

from openpmd_viewer import OpenPMDTimeSeries
from openpmd_viewer.addons import LpaDiagnostics

from visfbpic._mpl_config.laser_contour_cdict import laser_cmap
from visfbpic._mpl_config.cmaps import haline_cmap

plt.style.use(os.path.join(
    os.path.split(__file__)[0], '_mpl_config', 'style.mplstyle'))



def _find_nearest_idx(a, v):
    '''Return index of array a with value closest to v.'''
    a = np.asarray(a)
    return (np.abs(a - v)).argmin()



def animated_plasma_density(
    sup_dir,
    out_dir=None,
    r_roi=None,
    z_roi=(0, None),
    pol='y',
    n_max=None,
    relative_n_max=None,
    dpi=200,
    figsize=(8, 4.5),
    cmap="haline",
    z_units='window',
    interval=100
):
    '''Creates animated plasma density plot.
    
    Parameters
    ----------
    sup_dir : str
        Path to the simulation super directory. The simulations diags should be
        in here.
    out_dir : str, optional
        Path to output directory within which to save the animation. Defaults to
        None, which saves to <sup_dir>/analysis/plasma_density/
    r_roi : float, optional
        Radial coordinate of the region of interest in SI units. Defaults to
        None, which defines the ROI to be the whole domain.
    z_roi : tuple, optional
        z-directional coordinates of the region of interest in the form
        (z_low, z_up) in relative units (i.e. 0 is left boundary, 1 is right
        boundary). Defaults to (0, None), which selects the whole domain.
    pol : str, optional
        Polarisation of the laser to plot. Defaults to 'y'.
    n_max : float, optional
        The maximum value of the colour scale. Either this or relative_n_max
        must be supplied. Defaults to None.
    relative_n_max : float, optional
        The maximum value of the colour scale relative to the maximum number
        density at any time in the simulation. Either this or n_max must be
        supplied. Defaults to None.
    dpi : int, optional
        Dots per inch resolution. Changing this parameter may result in a bad
        plot layout. Defaults to 200.
    figsize : tuple, optional
        Figure size in the form (width, height). Changing this parameter may
        result in a bad plot layout. Defaults to (8, 4.5).
    cmap : str, optional
        Colour map for the number density plot. Either "haline" or "viridis".
        Defaults to "haline".
    z_units : str, optional
        Units of the z-axis. Either 'window' (default) or 'simulation'.
    interval : int, optional
        Interval between frames in ms. Governs the length of the animation.
        Defaults to 100.
    '''
    if n_max is None and relative_n_max is None:
        raise ValueError("Either n_max or relative_n_max must be supplied")
    elif n_max is not None and relative_n_max is not None:
        print("n_max and relative_n_max both supplied, defaulting to n_max")
        c_scale = "absolute"
    elif n_max is None and relative_n_max is not None:
        c_scale = "relative"
    elif n_max is not None and relative_n_max is None:
        c_scale = "absolute"
    
    if out_dir is None:
        out_dir = os.path.join(sup_dir, "analysis", "plasma_density")
        print("Output directory not specified")
        print(f"Defaulting to {out_dir}")
    if not os.path.isdir(out_dir):
        print(f"Making output directory at {out_dir}")
        os.makedirs(out_dir)
    
    sim_dir = os.path.join(sup_dir, "diags", "hdf5")
    
    # load diagnostics
    ts = OpenPMDTimeSeries(sim_dir)
    lpd = LpaDiagnostics(sim_dir)
    
    # get maximum number density in simulation
    print("Getting maximum number density")
    n_e_max = 0
    for i in tqdm(ts.iterations):
        rho_temp, _ = ts.get_field(iteration=i, field="rho")
        n_e_max_temp = np.abs(rho_temp).max() / e
        if n_e_max_temp > n_e_max:
            n_e_max = n_e_max_temp
    n_e_max_oom = int(np.floor(np.log10(n_e_max)))
    
    # get maximum number density for plot
    if c_scale == "relative":
        n_e_max_plot = n_e_max * relative_n_max
    elif c_scale == "absolute":
        n_e_max_plot = n_max
    n_e_max_plot_oom = int(np.floor(np.log10(n_e_max_plot)))
    n_e_max_plot /= 10.0 ** n_e_max_plot_oom
    
    # get initial info
    rho, info_rho = ts.get_field(iteration=0, field='rho')
    n_e = np.abs(rho) / e
    env, info_env = lpd.get_laser_envelope(iteration=0, pol=pol)
    
    # get indices of ROI
    if r_roi is None:
        r_low_idx = 0
        r_up_idx = None
    else:
        r_low_idx = _find_nearest_idx(info_rho.r, -1*r_roi)
        r_up_idx = _find_nearest_idx(info_rho.r, r_roi)
    if z_roi[0] == 0:
        z_low_idx = 0
    else:
        z_low_idx = int(info_rho.z.size * z_roi[0])
    if z_roi[1] is None:
        z_up_idx = None
    else:
        z_up_idx = int(info_rho.z.size * z_roi[1])
    
    # initial image extent
    extent = info_rho.imshow_extent
    if z_units == "window":
        extent[0] = 0
        if z_up_idx is None:
            extent[1] = info_rho.z[-1] - info_rho.z[z_low_idx]
        else:
            extent[1] = info_rho.z[z_up_idx] - info_rho.z[z_low_idx]
    else:
        extent[0] = info_rho.z[z_low_idx]
        if z_up_idx is None:
            extent[1] = info_rho.z[-1]
        else:
            extent[1] = info_rho.z[z_up_idx]
    extent[2] = info_rho.r[r_low_idx]
    if r_up_idx is None:
        extent[3] = info_rho.r[-1]
    else:
        extent[3] = info_rho.r[r_up_idx]
    
    # set up the figure
    fig = plt.figure(
        constrained_layout=False,
        figsize=figsize,
        dpi=dpi
    )
    spec = gridspec.GridSpec(
        ncols=2, nrows=1,
        left=0.1, right=0.9,
        bottom=0.1, top=0.9,
        wspace=0.05,
        width_ratios=[1, 0.03]
    )
    ax = fig.add_subplot(spec[0])
    cax = fig.add_subplot(spec[1])
    
    # get colour map
    if cmap == "haline":
        cmap = haline_cmap
    elif cmap == "viridis":
        cmap = "viridis"
    else:
        raise ValueError("cmap must be haline or viridis")
    
    # draw first frame
    im = ax.imshow(
        np.divide(n_e[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
                  10.0**n_e_max_plot_oom, dtype=np.float64),
        aspect='auto',
        extent=extent*1e6,
        vmin=0,
        vmax=n_e_max_plot,
        cmap=cmap
    )
    if z_units == "window":
        cs_z = info_rho.z[z_low_idx:z_up_idx] - info_rho.z[z_low_idx]
    else:
        cs_z = info_rho.z[z_low_idx:z_up_idx]
    cs = [ax.contour(
        cs_z*1e6,
        info_rho.r[r_low_idx:r_up_idx]*1e6,
        env[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
        levels=env.max()*np.linspace(1/np.e**2, 0.95, 4),
        cmap=laser_cmap
    )]
    fig.colorbar(im, cax=cax, extend='max')
    title_text = fig.text(
        0.5, 0.95,
        r'$t = {:.1f}$ ps'.format(ts.t[0]*1e12),
        va='center', ha='center', fontsize=11
    )
    ax.set_ylabel('$r$ ($\mathrm{\mu}$m)')
    if z_units == "window":
        ax.set_xlabel('$\zeta$ ($\mathrm{\mu}$m)')
    else:
        ax.set_xlabel('$z$ ($\mathrm{\mu}$m)')
    cax.set_ylabel(
        r'$n_e$ ($\times 10 ^ {{{}}}$ '.format(n_e_max_plot_oom) + 'm$^{-3}$)')
    
    print("Animating")
    def animate_plasma_density(i):
        '''Update the plot.'''
        print_string = str(i+1) + ' / ' + str(ts.iterations.size)
        print(print_string.ljust(20), end="\r", flush=True)
        
        # get data
        rho, info_rho = ts.get_field(iteration=ts.iterations[i], field='rho')
        n_e = np.abs(rho) / e
        env, info_env = lpd.get_laser_envelope(iteration=ts.iterations[i],
                                               pol=pol)
        
        # redraw data
        im.set_data(np.divide(
            n_e[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
            10.0**n_e_max_plot_oom, dtype=np.float64
        ))
        for c in cs[0].collections:
            c.remove()
        if z_units == "simulation":
            extent[0] = info_rho.z[z_low_idx]
            if z_up_idx is None:
                extent[1] = info_rho.z[-1]
            else:
                extent[1] = info_rho.z[z_up_idx]
            im.set_extent(extent*1e6)
        if z_units == "window":
            cs_z = info_rho.z[z_low_idx:z_up_idx] - info_rho.z[z_low_idx]
        else:
            cs_z = info_rho.z[z_low_idx:z_up_idx]
        cs[0] = ax.contour(
            cs_z*1e6,
            info_env.r[r_low_idx:r_up_idx]*1e6,
            env[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
            levels=env.max()*np.linspace(1/np.e**2, 0.95, 4),
            cmap=laser_cmap
        )
        
        # need to do this twice because of contours
        if z_units == "simulation":
            im.set_extent(extent*1e6)
        
        title_text.set_text(r'$t = {:.1f}$ ps'.format(ts.t[i]*1e12))
    
    ani = animation.FuncAnimation(
        fig, animate_plasma_density, range(0, ts.iterations.size),
        repeat=False, blit=False, interval=interval
    )
    
    ani.save(os.path.join(out_dir, "number_density_laser.mp4"))
    
    
    
def animated_plasma_density_and_info(
    sup_dir,
    n_file,
    z_min,
    z_max,
    z_points=1000,
    out_dir=None,
    r_roi=None,
    z_roi=(0, None),
    pol='y',
    n_max=None,
    relative_n_max=None,
    dpi=200,
    figsize=(8, 4.5),
    cmap="haline",
    z_units='window',
    interval=100,
):
    '''Creates animated plasma density and measured values plot.
    
    Parameters
    ----------
    sup_dir : str
        Path to the simulation super directory. The simulations diags should be
        in here.
    n_file : str
        Path to file containing number density info. See `_load_number_density`
        function (in this module) docstring.
    z_min, z_max : float
        Minimum and maximum values of z-coordinate for the plasma profile.
    z_points : int, optional
        Number of points to sample the number density profile along z.
    out_dir : str, optional
        Path to output directory within which to save the animation. Defaults to
        None, which saves to <sup_dir>/analysis/plasma_density/
    r_roi : float, optional
        Radial coordinate of the region of interest in SI units. Defaults to
        None, which defines the ROI to be the whole domain.
    z_roi : tuple, optional
        z-directional coordinates of the region of interest in the form
        (z_low, z_up) in relative units (i.e. 0 is left boundary, 1 is right
        boundary). Defaults to (0, None), which selects the whole domain.
    pol : str, optional
        Polarisation of the laser to plot. Defaults to 'y'.
    n_max : float, optional
        The maximum value of the colour scale. Either this or relative_n_max
        must be supplied. Defaults to None.
    relative_n_max : float, optional
        The maximum value of the colour scale relative to the maximum number
        density at any time in the simulation. Either this or n_max must be
        supplied. Defaults to None.
    dpi : int, optional
        Dots per inch resolution. Changing this parameter may result in a bad
        plot layout. Defaults to 200.
    figsize : tuple, optional
        Figure size in the form (width, height). Changing this parameter may
        result in a bad plot layout. Defaults to (8, 4.5).
    cmap : str, optional
        Colour map for the number density plot. Either "haline" or "viridis".
        Defaults to "haline".
    z_units : str, optional
        Units of the z-axis. Either 'window' (default) or 'simulation'.
    interval : int, optional
        Interval between frames in ms. Governs the length of the animation.
        Defaults to 100.
    '''
    if n_max is None and relative_n_max is None:
        raise ValueError("Either n_max or relative_n_max must be supplied")
    elif n_max is not None and relative_n_max is not None:
        print("n_max and relative_n_max both supplied, defaulting to n_max")
        c_scale = "absolute"
    elif n_max is None and relative_n_max is not None:
        c_scale = "relative"
    elif n_max is not None and relative_n_max is None:
        c_scale = "absolute"
    
    if out_dir is None:
        out_dir = os.path.join(sup_dir, "analysis", "complete")  
        print("Output directory not specified")
        print(f"Defaulting to {out_dir}")
    if not os.path.isdir(out_dir):
        print(f"Making output directory at {out_dir}")
        os.makedirs(out_dir)
    
    sim_dir = os.path.join(sup_dir, "diags", "hdf5")
    
    # load diagnostics
    ts = OpenPMDTimeSeries(sim_dir)
    lpd = LpaDiagnostics(sim_dir)
    
    # load number density distribution
    z_array, n_array = _load_number_density(n_file, z_min, z_max, z_points)
    n_array = np.array(n_array)
    n_profile_max_oom = int(np.floor(np.log10(n_array.max())))
    
    # getting measured values at every diagnostic dump
    print("Getting measured values")
    columns = [
        't', 'a0', 'lambda', 'w', 'ct', 'q', 'gamma', 'sigma gamma'
    ]
    ts_list = []  # simulation time
    a0_list = []  # laser a0
    lambda_list = []  # peak wavelength of laser spectrum
    w_list = []  # laser waist
    ct_list = []  # laser c tau
    q_list = []  # total charge of electrons matching criteria in input file
    gamma_list = []  # average gamma for electrons matching criteria in input file
    sigma_gamma_list = []  # std of gammas for electrons matching criteria in input file
    for t in tqdm(ts.t):
        ts_list.append(t)
        a0_list.append(lpd.get_a0(t=t, pol='y'))
        lambda_list.append(2*pi*c/lpd.get_main_frequency(t=t, pol='y'))
        w_list.append(lpd.get_laser_waist(t=t, pol='y'))
        ct_list.append(lpd.get_ctau(t=t, pol='y'))
        q_list.append(lpd.get_charge(species='electrons'))
        g_temp = lpd.get_mean_gamma(t=t, species='electrons')
        gamma_list.append(g_temp[0])
        sigma_gamma_list.append(g_temp[1])
    df_diag = pd.DataFrame(
        np.array(
            [ts_list, a0_list, lambda_list, w_list, ct_list,
            q_list, gamma_list, sigma_gamma_list]).T,
        columns=columns
    )
    
    # get maximum values of parameters
    print("Getting maximum values")
    n_e_max = 0
    q_e_max = 0
    gamma_max = 0
    gamma_plus_sigma_max = 0
    lambda_max = 0
    a0_max = 0
    w_max = 0
    ct_max = 0
    for i in tqdm(ts.iterations):
        rho_temp, _ = ts.get_field(iteration=i, field='rho')
        n_e_max_temp = np.abs(rho_temp).max()/e
        if n_e_max_temp > n_e_max:
            n_e_max = n_e_max_temp
    q_e_max = np.abs(df_diag['q']).max()
    gamma_max = df_diag['gamma'].max()
    gamma_plus_sigma_max = (df_diag['gamma'] + df_diag['sigma gamma']).max()
    lambda_max = df_diag['lambda'].max()
    a0_max = df_diag['a0'].max()
    w_max = df_diag['w'].max()
    ct_max = df_diag['ct'].max()

    # corresponding orders of magnitude
    n_e_max_oom = int(np.floor(np.log10(n_e_max)))
    try:
        q_e_max_oom = int(np.floor(np.log10(q_e_max)))
    except OverflowError:
        print("max charge is 0")
        q_e_max_oom = 0
    except ValueError:
        print("max charge is 0")
        q_e_max_oom = 0
    try:
        gamma_max_oom = int(np.floor(np.log10(gamma_max)))
    except OverflowError:
        gamma_max_oom = 0
    except ValueError:
        gamma_max_oom = 0
    try:
        gamma_plus_sigma_max_oom = int(np.floor(np.log10(gamma_plus_sigma_max)))
    except OverflowError:
        gamma_plus_sigma_max_oom = 0
    except ValueError:
        gamma_plus_sigma_max_oom = 0
    lambda_max_oom = int(np.floor(np.log10(lambda_max)))
    a0_max_oom = int(np.floor(np.log10(a0_max)))
    w_max_oom = int(np.floor(np.log10(w_max)))
    ct_max_oom = int(np.floor(np.log10(ct_max)))
    
    # get maximum number density for plot
    if c_scale == "relative":
        n_e_max_plot = n_e_max * relative_n_max
    elif c_scale == "absolute":
        n_e_max_plot = n_max
    n_e_max_plot_oom = int(np.floor(np.log10(n_e_max_plot)))
    n_e_max_plot /= 10.0 ** n_e_max_plot_oom
    
    # get initial info
    rho, info_rho = ts.get_field(iteration=0, field='rho')
    n_e = np.abs(rho) / e
    env, info_env = lpd.get_laser_envelope(iteration=0, pol=pol)
    
    # get indices of ROI
    if r_roi is None:
        r_low_idx = 0
        r_up_idx = None
    else:
        r_low_idx = _find_nearest_idx(info_rho.r, -1*r_roi)
        r_up_idx = _find_nearest_idx(info_rho.r, r_roi)
    if z_roi[0] == 0:
        z_low_idx = 0
    else:
        z_low_idx = int(info_rho.z.size * z_roi[0])
    if z_roi[1] is None:
        z_up_idx = None
    else:
        z_up_idx = int(info_rho.z.size * z_roi[1])
    
    # initial image extent
    extent = info_rho.imshow_extent
    if z_units == "window":
        extent[0] = 0
        if z_up_idx is None:
            extent[1] = info_rho.z[-1] - info_rho.z[z_low_idx]
        else:
            extent[1] = info_rho.z[z_up_idx] - info_rho.z[z_low_idx]
    else:
        extent[0] = info_rho.z[z_low_idx]
        if z_up_idx is None:
            extent[1] = info_rho.z[-1]
        else:
            extent[1] = info_rho.z[z_up_idx]
    extent[2] = info_rho.r[r_low_idx]
    if r_up_idx is None:
        extent[3] = info_rho.r[-1]
    else:
        extent[3] = info_rho.r[r_up_idx]
    
    # set up figure
    fig = plt.figure(
        constrained_layout=False,
        figsize=figsize,
        dpi=dpi
    )
    gs1 = gridspec.GridSpec(
        ncols=2, nrows=2,
        left=0.08, right=0.48,
        bottom=0.1, top=0.9,
        width_ratios=[1, 0.03],
        height_ratios=[0.25, 1],
        wspace=0.1,
        hspace=0.15
    )
    gs2 = gridspec.GridSpec(
        ncols=2, nrows=3,
        left=0.60, right=0.99,
        bottom=0.1, top=0.9,
        wspace=0.5
    )

    ax_density_profile = fig.add_subplot(gs1[0, :-1])
    ax_lwfa = fig.add_subplot(gs1[1, :-1])
    ax_lwfa_cb = fig.add_subplot(gs1[1, -1])
    ax_q = fig.add_subplot(gs2[0, 0])
    ax_gamma = fig.add_subplot(gs2[0, 1])
    ax_lambda = fig.add_subplot(gs2[1, 0])
    ax_a0 = fig.add_subplot(gs2[1, 1])
    ax_w = fig.add_subplot(gs2[2, 0])
    ax_ct = fig.add_subplot(gs2[2, 1])
    
    # get colour map
    if cmap == "haline":
        cmap = haline_cmap
    elif cmap == "viridis":
        cmap = "viridis"
    else:
        raise ValueError("cmap must be haline or viridis")
    
    # draw first frame
    ax_density_profile.plot(
        z_array*1e6,
        n_array/10**n_profile_max_oom,
        c='C1'
    )
    im_lwfa = ax_lwfa.imshow(
        np.divide(n_e[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
                  10.0**n_e_max_plot_oom, dtype=np.float64),
        aspect='auto',
        extent=extent*1e6,
        vmin=0,
        vmax=n_e_max_plot,
        cmap=cmap
    )
    if z_units == "window":
        cs_z = info_rho.z[z_low_idx:z_up_idx] - info_rho.z[z_low_idx]
    else:
        cs_z = info_rho.z[z_low_idx:z_up_idx]
    cs_lwfa = [
        ax_lwfa.contour(
            cs_z*1e6,
            info_env.r[r_low_idx:r_up_idx]*1e6,
            env[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
            levels=env.max()*np.linspace(1/np.e**2, 0.95, 4),
            cmap=laser_cmap
        )
    ]
    ax_q_line, = ax_q.plot(
        df_diag['t']*1e12,
        np.abs(df_diag['q'])/10**q_e_max_oom,
        c='C1'
    )
    ax_q_scat, = ax_q.plot(
        df_diag['t']*1e12,
        np.abs(df_diag['q'])/10**q_e_max_oom,
        c='C1',
        marker='o'
    )
    ax_gamma_line, = ax_gamma.plot(
        df_diag['t']*1e12,
        df_diag['gamma']/10**gamma_plus_sigma_max_oom,
        c='C1',
    )
    ax_gamma_scat, = ax_gamma.plot(
        df_diag['t']*1e12,
        df_diag['gamma']/10**gamma_plus_sigma_max_oom,
        c='C1',
        marker='o'
    )
    ax_lambda_line, = ax_lambda.plot(
        df_diag['t']*1e12,
        df_diag['lambda']/10**lambda_max_oom,
        c='C0',
    )
    ax_lambda_scat, = ax_lambda.plot(
        df_diag['t']*1e12,
        df_diag['lambda']/10**lambda_max_oom,
        c='C0',
        marker='o'
    )
    ax_a0_line, = ax_a0.plot(
        df_diag['t']*1e12,
        df_diag['a0']/10**a0_max_oom,
        c='C0',
    )
    ax_a0_scat, = ax_a0.plot(
        df_diag['t']*1e12,
        df_diag['a0']/10**a0_max_oom,
        marker='o',
        c='C0'
    )
    ax_w_line, = ax_w.plot(
        df_diag['t']*1e12,
        df_diag['w']/10**w_max_oom,
        c='C0'
    )
    ax_w_scat, = ax_w.plot(
        df_diag['t']*1e12,
        df_diag['w']/10**w_max_oom,
        marker='o',
        c='C0'
    )
    ax_ct_line, = ax_ct.plot(
        df_diag['t']*1e12,
        df_diag['ct']/10**ct_max_oom,
        c='C0'
    )
    ax_ct_scat, = ax_ct.plot(
        df_diag['t']*1e12,
        df_diag['ct']/10**ct_max_oom,
        marker='o',
        c='C0'
    )
    ax_gamma.fill_between(
        df_diag['t']*1e12,
        df_diag['gamma']/10**gamma_plus_sigma_max_oom-\
        df_diag['sigma gamma']/10**gamma_plus_sigma_max_oom,
        df_diag['gamma']/10**gamma_plus_sigma_max_oom+\
        df_diag['sigma gamma']/10**gamma_plus_sigma_max_oom,
        color='C1',
        alpha=0.3,
        edgecolor=None
    )
    
    fig.colorbar(im_lwfa, cax=ax_lwfa_cb, extend='max')
    
    # remove unneccessary tick labels
    ax_q.set_xticklabels([])
    ax_gamma.set_xticklabels([])
    ax_lambda.set_xticklabels([])
    ax_a0.set_xticklabels([])
    
    # change tick positions
    ax_density_profile.xaxis.tick_top()
    ax_density_profile.xaxis.set_label_position('top')
    
    # set precision
    y_major_formatter = FormatStrFormatter('%.2f')
    ax_q.yaxis.set_major_formatter(y_major_formatter)
    ax_lambda.yaxis.set_major_formatter(y_major_formatter)
    ax_w.yaxis.set_major_formatter(y_major_formatter)
    ax_gamma.yaxis.set_major_formatter(y_major_formatter)
    ax_a0.yaxis.set_major_formatter(y_major_formatter)
    ax_ct.yaxis.set_major_formatter(y_major_formatter)

    # set axis labels
    ax_density_profile.set_xlabel('$z$ ($\mathrm{\mu}$m)')
    ax_density_profile.set_ylabel(r'$n_e$ ($\times 10 ^ {{{}}}$'.format(
        n_profile_max_oom-6) + ' cm$^{-3}$)')
    if z_units == "window":
        ax_lwfa.set_xlabel('$\zeta$ ($\mathrm{\mu}$m)')
    else:
        ax_lwfa.set_xlabel('$z$ ($\mathrm{\mu}$m)')
    ax_lwfa.set_ylabel('$r$ ($\mathrm{\mu}$m)')
    ax_lwfa_cb.set_title('$n_e$\n' + r'($\times 10 ^ {{{}}}$ '.format(
        n_e_max_plot_oom-6) + 'cm$^{-3}$)')
    ax_q.set_ylabel(r'$|q_e|$ ($\times 10 ^ {{{}}}$ '.format(q_e_max_oom) + 'C)')
    ax_gamma.set_ylabel(r'$\hat{\gamma}$' + r' ($\times 10 ^ {{{}}}$'.format(
        gamma_max_oom) + ')')
    ax_lambda.set_ylabel(r'$\lambda_\mathrm{peak}$ ' + \
                        r'($\times 10 ^ {{{}}}$ '.format(lambda_max_oom) + 'm)')
    ax_a0.set_ylabel(r'$a_0$ ($\times 10 ^ {{{}}}$'.format(a0_max_oom) + ')')
    ax_w.set_ylabel(r'$w_l$ ($\times 10 ^ {{{}}}$ '.format(w_max_oom) + 'm)')
    ax_w.set_xlabel(r'$t$ (ps)')
    ax_ct.set_ylabel(r'$c \tau_l$ ($\times 10 ^ {{{}}}$ '.format(ct_max_oom) +\
        'm)')
    ax_ct.set_xlabel(r'$t$ (ps)')

    fig.align_ylabels([ax_q, ax_lambda, ax_w])
    fig.align_ylabels([ax_gamma, ax_a0, ax_ct])

    t_text = fig.text(
        ax_q.get_position().x0+(ax_gamma.get_position().x1-ax_q.get_position().x0)/2,
        ax_q.get_position().y1+(1-ax_q.get_position().y1)/2,
        r'$t = {:.1f}$ ps'.format(ts.t[0]*1e12),
        va='center', ha='center', fontsize=11
    )

    fig.canvas.draw()
    ax_density_profile.set_ylim(0, ax_density_profile.get_ylim()[1])
    ax_density_profile.fill_between(
        [info_rho.z[0]*1e6, info_rho.z[-1]*1e6],
        [0, 0],
        [ax_density_profile.get_ylim()[1], ax_density_profile.get_ylim()[1]],
        color='C1',
        alpha=0.5,
        edgecolor=None
    )
    
    def animate_plasma_density_and_info(i):
        # global cs_lwfa
        print_string = str(i+1) + ' / ' + str(ts.iterations.size)
        print(print_string.ljust(20), end="\r", flush=True)
        
        # get data
        rho, info_rho = ts.get_field(iteration=ts.iterations[i], field='rho')
        n_e = np.abs(rho) / e
        env, info_env = lpd.get_laser_envelope(iteration=ts.iterations[i], pol='y')
        
        # redraw the data
        im_lwfa.set_data(np.divide(n_e[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
                                10.0**n_e_max_plot_oom, dtype=np.float64))
        
        ax_q_scat.set_data(
            df_diag['t'][i]*1e12,
            np.abs(df_diag['q'][i])/10**q_e_max_oom
        )
        ax_q_line.set_data(
            df_diag['t'][:i+1]*1e12,
            np.abs(df_diag['q'][:i+1])/10**q_e_max_oom
        )
        
        ax_gamma_scat.set_data(
            df_diag['t'][i]*1e12,
            df_diag['gamma'][i]/10**gamma_plus_sigma_max_oom
        )
        ax_gamma_line.set_data(
            df_diag['t'][:i+1]*1e12,
            df_diag['gamma'][:i+1]/10**gamma_plus_sigma_max_oom
        )
        
        ax_lambda_scat.set_data(
            df_diag['t'][i]*1e12,
            df_diag['lambda'][i]/10**lambda_max_oom
        )
        ax_lambda_line.set_data(
            df_diag['t'][:i+1]*1e12,
            df_diag['lambda'][:i+1]/10**lambda_max_oom
        )
        
        ax_a0_scat.set_data(
            df_diag['t'][i]*1e12,
            df_diag['a0'][i]/10**a0_max_oom
        )
        ax_a0_line.set_data(
            df_diag['t'][:i+1]*1e12,
            df_diag['a0'][:i+1]/10**a0_max_oom
        )
        
        ax_w_scat.set_data(
            df_diag['t'][i]*1e12,
            df_diag['w'][i]/10**w_max_oom
        )
        ax_w_line.set_data(
            df_diag['t'][:i+1]*1e12,
            df_diag['w'][:i+1]/10**w_max_oom
        )
        
        ax_ct_scat.set_data(
            df_diag['t'][i]*1e12,
            df_diag['ct'][i]/10**ct_max_oom
        )
        ax_ct_line.set_data(
            df_diag['t'][:i+1]*1e12,
            df_diag['ct'][:i+1]/10**ct_max_oom
        )
        
        if z_units == "simulation":
            extent[0] = info_rho.z[z_low_idx]
            if z_up_idx is None:
                extent[1] = info_rho.z[-1]
            else:
                extent[1] = info_rho.z[z_up_idx]
            im_lwfa.set_extent(extent*1e6)        
        
        t_text.set_text(r'$t = {:.1f}$ ps'.format(ts.t[i]*1e12))
        
        ax_density_profile.collections.clear()
        ax_density_profile.fill_between(
            [info_rho.z[0]*1e6, info_rho.z[-1]*1e6]       ,     
            [0, 0],
            [ax_density_profile.get_ylim()[1], ax_density_profile.get_ylim()[1]],
            color='C1',
            alpha=0.5,
            edgecolor=None
        )
        
        ax_gamma.collections.clear()
        if i > 0:
            ax_gamma.fill_between(
                df_diag['t'][:i+1]*1e12,
                df_diag['gamma'][:i+1]/10**gamma_plus_sigma_max_oom-\
                df_diag['sigma gamma'][:i+1]/10**gamma_plus_sigma_max_oom,
                df_diag['gamma'][:i+1]/10**gamma_plus_sigma_max_oom+\
                df_diag['sigma gamma'][:i+1]/10**gamma_plus_sigma_max_oom,
                color='C1',
                alpha=0.3,
                edgecolor=None
            )
        
        if z_units == "window":
            cs_z = info_rho.z[z_low_idx:z_up_idx] - info_rho.z[z_low_idx]
        else:
            cs_z = info_rho.z[z_low_idx:z_up_idx]
        for c in cs_lwfa[0].collections:
            c.remove()
        cs_lwfa[0] = ax_lwfa.contour(
            cs_z*1e6,
            info_env.r[r_low_idx:r_up_idx]*1e6,
            env[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
            levels=env.max()*np.linspace(1/np.e**2, 0.95, 4),
            cmap=laser_cmap
        )
        im_lwfa.set_extent(extent*1e6)
        
        return cs_lwfa
    
    print("Animating")
    ani = animation.FuncAnimation(
        fig, animate_plasma_density_and_info, range(0, ts.iterations.size),
        repeat=False, blit=False, interval=interval
    )
    
    ani.save(os.path.join(out_dir, "complete.mp4"))
    
    
    
def animated_plasma_density_and_info_compare(
    sup_dir_1,
    sup_dir_2,
    n_file_1,
    n_file_2,
    z_min,
    z_max,
    out_dir,
    z_points=1000,
    r_roi=None,
    z_roi=(0, None),
    pol_1='y',
    pol_2='y',
    n_max=None,
    relative_n_max=None,
    dpi=200,
    figsize=(8, 4.5),
    cmap="haline",
    interval=100,
    n_procs=1,
):
    '''
    Creates animated plasma density and measured values plots for two
    simulations.
    
    The parameter names ending in _1 and _2 are for the two simulations to be
    compared.
    
    No checks are performed to see if the two simulations are comparable.
    
    Parameters
    ----------
    sup_dir_1, sup_dir_2 : str
        Path to the simulation super directories. The simulation diags should be
        in here.
    n_file_1, n_file_2 : str
        Path to files containing the number density descriptors. Can be `.py`
        files containing a `number_density` function (see
        `animated_plasma_density_and_info` docstring in this module) or `.csv`
        files.
    z_min, z_max : float
        Minimum and maximum values of z-coordinate for the plasma profiles.
    out_dir : str
        Path to output directory within which to save the animation.
    z_points : int, optional
        Number of points to sample the number density profile along z.
    r_roi : float, optional
        Radial coordinate of the region of interest in SI units. Defaults to
        None, which defines the ROI to be the whole domain.
    z_roi : tuple, optional
        z-directional coordinates of the region of interest in the form
        (z_low, z_up) in relative units (i.e. 0 is left boundary, 1 is right
        boundary). Defaults to (0, None), which selects the whole domain.
    pol_1, pol_2 : str, optional
        Polarisation of the laser. Defaults to 'y'.
    n_max : float, optional
        The maximum value of the colour scale. Either this or relative_n_max
        must be supplied. Defaults to None.
    relative_n_max : float, optional
        The maximum value of the colour scale relative to the maximum number
        density at any time in the simulation. Either this or n_max must be
        supplied. Defaults to None.
    dpi : int, optional
        Dots per inch resolution. Changing this parameter may result in a bad
        plot layout. Defaults to 200.
    figsize : tuple, optional
        Figure size in the form (width, height). Changing this parameter may
        result in a bad plot layout. Defaults to (8, 4.5).
    cmap : str, optional
        Colour map for the number density plot. Either "haline" or "viridis".
        Defaults to "haline".
    interval : int, optional
        Interval between frames in ms. Governs the length of the animation.
        Defaults to 100.
    '''
    if n_max is None and relative_n_max is None:
        raise ValueError("Either n_max or relative_n_max must be supplied")
    elif n_max is not None and relative_n_max is not None:
        print("n_max and relative_n_max both supplied, defaulting to n_max")
        c_scale = "absolute"
    elif n_max is None and relative_n_max is not None:
        c_scale = "relative"
    elif n_max is not None and relative_n_max is None:
        c_scale = "absolute"
        
    z_units = "window"
    
    sim_dir_1 = os.path.join(sup_dir_1, "diags", "hdf5")
    sim_dir_2 = os.path.join(sup_dir_2, "diags", "hdf5")
    
    # load diagnostics
    ts_1 = OpenPMDTimeSeries(sim_dir_1)
    ts_2 = OpenPMDTimeSeries(sim_dir_2)
    lpd_1 = LpaDiagnostics(sim_dir_1)
    lpd_2 = LpaDiagnostics(sim_dir_2)
    
    # generate or load number density distributions
    z_array_1, n_array_1 = _load_number_density(
        n_file_1, z_min, z_max, z_points
    )
    z_array_2, n_array_2 = _load_number_density(
        n_file_2, z_min, z_max, z_points
    )        
    n_profile_1_max_oom = int(np.floor(np.log10(n_array_1.max())))
    n_profile_2_max_oom = int(np.floor(np.log10(n_array_2.max())))
    n_profile_max_oom = max((n_profile_1_max_oom, n_profile_2_max_oom))
    
    # getting measured values at every diagnostic dump
    columns = [
        't', 'a0', 'lambda', 'w', 'ct', 'q', 'gamma', 'sigma gamma'
    ]
    ts_list_1, ts_list_2 = [], []  # simulation time
    a0_list_1, a0_list_2 = [], []  # laser a0
    lambda_list_1, lambda_list_2 = [], []  # peak wavelength of laser spectrum
    w_list_1, w_list_2 = [], []  # laser waist
    ct_list_1, ct_list_2 = [], []  # laser c tau
    q_list_1, q_list_2 = [], []  # total charge of electrons matching criteria
                                 # in input file
    gamma_list_1, gamma_list_2 = [], []  # average gamma for electrons matching
                                         # criteria in input file
    sigma_gamma_list_1, sigma_gamma_list_2 = [], []  # std of gammas for
                                                     # electrons matching
                                                     # criteria in input file
    print("Getting measured values 1/2")
    for t in tqdm(ts_1.t):
        ts_list_1.append(t)
        a0_list_1.append(lpd_1.get_a0(t=t, pol='y'))
        lambda_list_1.append(2*pi*c/lpd_1.get_main_frequency(t=t, pol='y'))
        w_list_1.append(lpd_1.get_laser_waist(t=t, pol='y'))
        ct_list_1.append(lpd_1.get_ctau(t=t, pol='y'))
        q_list_1.append(lpd_1.get_charge(species='electrons'))
        g_temp_1 = lpd_1.get_mean_gamma(t=t, species='electrons')
        gamma_list_1.append(g_temp_1[0])
        sigma_gamma_list_1.append(g_temp_1[1])
    df_diag_1 = pd.DataFrame(
        np.array(
            [ts_list_1, a0_list_1, lambda_list_1, w_list_1, ct_list_1,
            q_list_1, gamma_list_1, sigma_gamma_list_1]).T,
        columns=columns
    )
    print("Getting measured values 2/2")
    for t in tqdm(ts_2.t):
        ts_list_2.append(t)
        a0_list_2.append(lpd_2.get_a0(t=t, pol='y'))
        lambda_list_2.append(2*pi*c/lpd_2.get_main_frequency(t=t, pol='y'))
        w_list_2.append(lpd_2.get_laser_waist(t=t, pol='y'))
        ct_list_2.append(lpd_2.get_ctau(t=t, pol='y'))
        q_list_2.append(lpd_2.get_charge(species='electrons'))
        g_temp_2 = lpd_2.get_mean_gamma(t=t, species='electrons')
        gamma_list_2.append(g_temp_2[0])
        sigma_gamma_list_2.append(g_temp_2[1])
    df_diag_2 = pd.DataFrame(
        np.array(
            [ts_list_2, a0_list_2, lambda_list_2, w_list_2, ct_list_2,
            q_list_2, gamma_list_2, sigma_gamma_list_2]).T,
        columns=columns
    )
    
    # getting maximum values of parameters
    print("Getting maximum values")
    n_e_max = 0
    q_e_max = 0
    gamma_max = 0
    gamma_plus_sigma_max = 0
    lambda_max = 0
    a0_max = 0
    w_max = 0
    ct_max = 0
    print('1/2')
    for i in tqdm(ts_1.iterations):
        rho_temp, _ = ts_1.get_field(iteration=i, field='rho')
        n_e_max_temp = np.abs(rho_temp).max() / e
        if n_e_max_temp > n_e_max:
            n_e_max = n_e_max_temp
    print('2/2')
    for i in tqdm(ts_2.iterations):
        rho_temp, _ = ts_2.get_field(iteration=i, field='rho')
        n_e_max_temp = np.abs(rho_temp).max() / e
        if n_e_max_temp > n_e_max:
            n_e_max = n_e_max_temp
    q_e_max = max([np.abs(df_diag_1['q']).max(), np.abs(df_diag_2['q']).max()])
    gamma_max = max([df_diag_1['gamma'].max(), df_diag_2['gamma'].max()])
    gamma_plus_sigma_max = max([
        (df_diag_1['gamma'] + df_diag_1['sigma gamma']).max(),
        (df_diag_2['gamma'] + df_diag_1['sigma gamma']).max()
    ])
    lambda_max = max([df_diag_1['lambda'].max(), df_diag_2['lambda'].max()])
    a0_max = max([df_diag_1['a0'].max(), df_diag_2['a0'].max()])
    w_max = max([df_diag_1['w'].max(), df_diag_2['w'].max()])
    ct_max = max([df_diag_1['ct'].max(), df_diag_2['ct'].max()])
    
    # corresponding orders of magnitude
    n_e_max_oom = int(np.floor(np.log10(n_e_max)))
    try:
        q_e_max_oom = int(np.floor(np.log10(q_e_max)))
    except OverflowError:
        print("max charge is 0")
        q_e_max_oom = 0
    except ValueError:
        print("max charge is 0")
        q_e_max_oom = 0
    try:
        gamma_max_oom = int(np.floor(np.log10(gamma_max)))
    except OverflowError:
        gamma_max_oom = 0
    except ValueError:
        gamma_max_oom = 0
    try:
        gamma_plus_sigma_max_oom = int(np.floor(np.log10(gamma_plus_sigma_max)))
    except OverflowError:
        gamma_plus_sigma_max_oom = 0
    except ValueError:
        gamma_plus_sigma_max_oom = 0
    lambda_max_oom = int(np.floor(np.log10(lambda_max)))
    a0_max_oom = int(np.floor(np.log10(a0_max)))
    w_max_oom = int(np.floor(np.log10(w_max)))
    ct_max_oom = int(np.floor(np.log10(ct_max)))
    
    # get maximum number density for plot
    if c_scale == "relative":
        n_e_max_plot = n_e_max * relative_n_max
    elif c_scale == "absolute":
        n_e_max_plot = n_max
    n_e_max_plot_oom = int(np.floor(np.log10(n_e_max_plot)))
    n_e_max_plot /= 10.0 ** n_e_max_plot_oom
    
    # get initial info
    rho_1, info_rho_1 = ts_1.get_field(iteration=0, field='rho')
    rho_2, info_rho_2 = ts_2.get_field(iteration=0, field='rho')
    n_e_1 = np.abs(rho_1) / e
    n_e_2 = np.abs(rho_2) / e
    env_1, info_env_1 = lpd_1.get_laser_envelope(iteration=0, pol=pol_1)
    env_2, info_env_2 = lpd_2.get_laser_envelope(iteration=0, pol=pol_2)
    
    # get indices of ROI
    if r_roi is None:
        r_low_idx = 0
        r_up_idx = None
    else:
        r_low_idx = _find_nearest_idx(info_rho_1.r, -1*r_roi)
        r_up_idx = _find_nearest_idx(info_rho_1.r, r_roi)
    if z_roi[0] == 0:
        z_low_idx = 0
    else:
        z_low_idx = int(info_rho_1.z.size * z_roi[0])
    if z_roi[1] is None:
        z_up_idx = None
    else:
        z_up_idx = int(info_rho_1.z.size * z_roi[1])
    
    # initial image extent
    extent = info_rho_1.imshow_extent
    if z_units == "window":
        extent[0] = 0
        if z_up_idx is None:
            extent[1] = info_rho_1.z[-1] - info_rho_1.z[z_low_idx]
        else:
            extent[1] = info_rho_1.z[z_up_idx] - info_rho_1.z[z_low_idx]
    else:
        extent[0] = info_rho_1.z[z_low_idx]
        if z_up_idx is None:
            extent[1] = info_rho_1.z[-1]
        else:
            extent[1] = info_rho_1.z[z_up_idx]
    extent[2] = info_rho_1.r[r_low_idx]
    if r_up_idx is None:
        extent[3] = info_rho_1.r[-1]
    else:
        extent[3] = info_rho_1.r[r_up_idx]
    
    # set up figure
    fig = plt.figure(
        constrained_layout=False,
        figsize=figsize,
        dpi=dpi
    )
    gs1 = gridspec.GridSpec(
        ncols=2, nrows=2,
        left=0.08, right=0.48,
        bottom=0.1, top=0.9,
        width_ratios=[1, 0.03],
        height_ratios=[0.25, 1],
        wspace=0.1,
        hspace=0.15
    )
    gs2 = gridspec.GridSpec(
        ncols=2, nrows=3,
        left=0.60, right=0.99,
        bottom=0.1, top=0.9,
        wspace=0.5
    )

    ax_density_profile = fig.add_subplot(gs1[0, :-1])
    ax_lwfa = fig.add_subplot(gs1[1, :-1])
    ax_lwfa_cb = fig.add_subplot(gs1[1, -1])
    ax_q = fig.add_subplot(gs2[0, 0])
    ax_gamma = fig.add_subplot(gs2[0, 1])
    ax_lambda = fig.add_subplot(gs2[1, 0])
    ax_a0 = fig.add_subplot(gs2[1, 1])
    ax_w = fig.add_subplot(gs2[2, 0])
    ax_ct = fig.add_subplot(gs2[2, 1])
    
    # custom 'legend'
    leg_x, leg_y = 0.7, 0.03
    leg_width, leg_height = 0.3, 0.05
    ax_leg_1 = fig.add_axes((
        ax_lwfa.get_position().x0 + leg_x * ax_lwfa.get_position().width,
        ax_lwfa.get_position().y0 + (1 - leg_y - leg_height) * ax_lwfa.get_position().height,
        leg_width * ax_lwfa.get_position().width,
        leg_height * ax_lwfa.get_position().height
    ))
    ax_leg_2 = fig.add_axes((
        ax_lwfa.get_position().x0 + leg_x * ax_lwfa.get_position().width,
        ax_lwfa.get_position().y0 + leg_y * ax_lwfa.get_position().height,
        leg_width * ax_lwfa.get_position().width,
        leg_height * ax_lwfa.get_position().height
    ))
    fancybox_1 = mpatches.FancyBboxPatch(
        (0.07, 0.07),
        0.9, 0.9,
        facecolor='w',
        edgecolor='None',
        alpha=0.3,
        boxstyle="round,pad=0.02",
        mutation_scale=0.2
    )
    fancybox_2 = mpatches.FancyBboxPatch(
        (0.07, 0.07),
        0.9, 0.9,
        facecolor='w',
        edgecolor='None',
        alpha=0.3,
        boxstyle="round,pad=0.02",
        mutation_scale=0.2
    )
    ax_leg_1.add_patch(fancybox_1)
    ax_leg_2.add_patch(fancybox_2)
    pos_y_1 = 0.5
    pos_x_1 = 4/10
    pos_x_2 = 9/10
    pos_x_1_1 = 1/10
    pos_x_1_2 = 4/10
    pos_x_2_1 = 6/10
    pos_x_2_2 = 9/10
    ms = 4
    ax_leg_1.plot(
        [pos_x_1],
        [pos_y_1],
        ls='',
        marker='s',
        ms=ms,
        c='C1'
    )
    ax_leg_1.plot(
        [pos_x_1_1, pos_x_1_2],
        [pos_y_1, pos_y_1],
        ls='-',
        c='C1'
    )
    ax_leg_1.plot(
        [pos_x_2],
        [pos_y_1],
        ls='',
        marker='s',
        ms=ms,
        c='C0'
    )
    ax_leg_1.plot(
        [pos_x_2_1, pos_x_2_2],
        [pos_y_1, pos_y_1],
        ls='-',
        c='C0'
    )
    ax_leg_2.plot(
        [pos_x_1],
        [pos_y_1],
        ls='',
        marker='o',
        ms=ms,
        c='C3'
    )
    ax_leg_2.plot(
        [pos_x_1_1, pos_x_1_2],
        [pos_y_1, pos_y_1],
        ls='--',
        c='C3'
    )
    ax_leg_2.plot(
        [pos_x_2],
        [pos_y_1],
        ls='',
        marker='o',
        ms=ms,
        c='C2'
    )
    ax_leg_2.plot(
        [pos_x_2_1, pos_x_2_2],
        [pos_y_1, pos_y_1],
        ls='--',
        c='C2'
    )
    ax_leg_1.set_xlim(0, 1)
    ax_leg_1.set_ylim(0, 1)
    ax_leg_2.set_xlim(0, 1)
    ax_leg_2.set_ylim(0, 1)
    ax_leg_1.get_xaxis().set_visible(False)
    ax_leg_1.get_yaxis().set_visible(False)
    ax_leg_2.get_xaxis().set_visible(False)
    ax_leg_2.get_yaxis().set_visible(False)
    ax_leg_1.axis('off')
    ax_leg_2.axis('off')
    
    # get colour map
    if cmap == "haline":
        cmap = haline_cmap
    elif cmap == "viridis":
        cmap = "viridis"
    else:
        raise ValueError("cmap must be haline or viridis")
    
    # draw first frame
    ax_density_profile.plot(
        z_array_1*1e6,
        n_array_1/10**n_profile_max_oom,
        c='C1'
    )
    ax_density_profile.plot(
        z_array_2*1e6,
        n_array_2/10**n_profile_max_oom,
        c='C3',
        ls='--'
    )
    n_e_im = _glue_images(
        np.divide(n_e_1[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
                  10.0**n_e_max_plot_oom, dtype=np.float64),
        np.divide(n_e_2[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
                  10.0**n_e_max_plot_oom, dtype=np.float64)
    )
    im_lwfa = ax_lwfa.imshow(
        n_e_im,
        aspect='auto',
        extent=extent*1e6,
        vmin=0,
        vmax=n_e_max_plot,
        cmap=cmap
    )
    env_im = _glue_images(
        env_1[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
        env_2[r_low_idx:r_up_idx, z_low_idx:z_up_idx]
    )
    if z_units == "window":
        cs_z = info_rho_1.z[z_low_idx:z_up_idx] - info_rho_1.z[z_low_idx]
    else:
        cs_z = info_rho_1.z[z_low_idx:z_up_idx]
    cs_lwfa = [
        ax_lwfa.contour(
            cs_z*1e6,
            info_env_1.r[r_low_idx:r_up_idx]*1e6,
            env_im,
            levels=env_im.max()*np.linspace(1/np.e**2, 0.95, 4),
            cmap=laser_cmap
        )
    ]
    
    if z_units == "window":
        if z_up_idx is None:
            div_z = [0, info_rho_1.z[-1]]
        else:
            div_z = [
                0, info_rho_1.z[z_up_idx] - info_rho_1.z[z_low_idx]
            ]
    else:
        div_z = [
            info_rho_1.z[z_low_idx], info_rho_1.z[z_up_idx]
        ]
    div_line, = ax_lwfa.plot(
        div_z,
        [0, 0],
        c='white',
        ls='--',
        alpha=0.5
    )
    
    ms = 4
    ax_q_line_1, = ax_q.plot(
        df_diag_1['t']*1e12,
        np.abs(df_diag_1['q'])/10**q_e_max_oom,
        c='C1'
    )
    ax_q_line_2, = ax_q.plot(
        df_diag_2['t']*1e12,
        np.abs(df_diag_2['q'])/10**q_e_max_oom,
        c='C3',
        ls='--'
    )
    ax_q_scat_1, = ax_q.plot(
        df_diag_1['t']*1e12,
        np.abs(df_diag_1['q']/10**q_e_max_oom),
        c='C1',
        marker='s'
    )
    ax_q_scat_2, = ax_q.plot(
        df_diag_2['t']*1e12,
        np.abs(df_diag_2['q']/10**q_e_max_oom),
        c='C3',
        marker='o',
        ms=ms
    )
    ax_gamma_line_1, = ax_gamma.plot(
        df_diag_1['t']*1e12,
        df_diag_1['gamma']/10**gamma_plus_sigma_max_oom,
        c='C1'
    )
    ax_gamma_line_2, = ax_gamma.plot(
        df_diag_2['t']*1e12,
        df_diag_2['gamma']/10**gamma_plus_sigma_max_oom,
        c='C3',
        ls='--'
    )
    ax_gamma_scat_1, = ax_gamma.plot(
        df_diag_1['t']*1e12,
        df_diag_1['gamma']/10**gamma_plus_sigma_max_oom,
        c='C1',
        marker='s'
    )
    ax_gamma_scat_2, = ax_gamma.plot(
        df_diag_2['t']*1e12,
        df_diag_2['gamma']/10**gamma_plus_sigma_max_oom,
        c='C3',
        marker='o',
        ms=ms
    )
    ax_lambda_line_1, = ax_lambda.plot(
        df_diag_1['t']*1e12,
        df_diag_1['lambda']/10**lambda_max_oom,
        c='C0',
    )
    ax_lambda_line_2, = ax_lambda.plot(
        df_diag_2['t']*1e12,
        df_diag_2['lambda']/10**lambda_max_oom,
        c='C2',
        ls='--'
    )
    ax_lambda_scat_1, = ax_lambda.plot(
        df_diag_1['t']*1e12,
        df_diag_1['lambda']/10**lambda_max_oom,
        c='C0',
        marker='s'
    )
    ax_lambda_scat_2, = ax_lambda.plot(
        df_diag_2['t']*1e12,
        df_diag_2['lambda']/10**lambda_max_oom,
        c='C2',
        marker='o',
        ms=ms
    )
    ax_a0_line_1, = ax_a0.plot(
        df_diag_1['t']*1e12,
        df_diag_1['a0']/10**a0_max_oom,
        c='C0',
    )
    ax_a0_line_2, = ax_a0.plot(
        df_diag_2['t']*1e12,
        df_diag_2['a0']/10**a0_max_oom,
        c='C2',
        ls='--'
    )
    ax_a0_scat_1, = ax_a0.plot(
        df_diag_1['t']*1e12,
        df_diag_1['a0']/10**a0_max_oom,
        marker='s',
        c='C0'
    )
    ax_a0_scat_2, = ax_a0.plot(
        df_diag_2['t']*1e12,
        df_diag_2['a0']/10**a0_max_oom,
        marker='o',
        c='C2',
        ms=ms
    )
    ax_w_line_1, = ax_w.plot(
        df_diag_1['t']*1e12,
        df_diag_1['w']/10**w_max_oom,
        c='C0'
    )
    ax_w_line_2, = ax_w.plot(
        df_diag_2['t']*1e12,
        df_diag_2['w']/10**w_max_oom,
        c='C2',
        ls='--'
    )
    ax_w_scat_1, = ax_w.plot(
        df_diag_1['t']*1e12,
        df_diag_1['w']/10**w_max_oom,
        marker='s',
        c='C0'
    )
    ax_w_scat_2, = ax_w.plot(
        df_diag_2['t']*1e12,
        df_diag_2['w']/10**w_max_oom,
        marker='o',
        c='C2',
        ms=ms
    )
    ax_ct_line_1, = ax_ct.plot(
        df_diag_1['t']*1e12,
        df_diag_1['ct']/10**ct_max_oom,
        c='C0'
    )
    ax_ct_line_2, = ax_ct.plot(
        df_diag_2['t']*1e12,
        df_diag_2['ct']/10**ct_max_oom,
        c='C2',
        ls='--'
    )
    ax_ct_scat_1, = ax_ct.plot(
        df_diag_1['t']*1e12,
        df_diag_1['ct']/10**ct_max_oom,
        marker='s',
        c='C0'
    )
    ax_ct_scat_2, = ax_ct.plot(
        df_diag_2['t']*1e12,
        df_diag_2['ct']/10**ct_max_oom,
        marker='o',
        c='C2',
        ms=ms
    )
    ax_gamma.fill_between(
        df_diag_1['t']*1e12,
        df_diag_1['gamma']/10**gamma_plus_sigma_max_oom-\
        df_diag_1['sigma gamma']/10**gamma_plus_sigma_max_oom,
        df_diag_1['gamma']/10**gamma_plus_sigma_max_oom+\
        df_diag_1['sigma gamma']/10**gamma_plus_sigma_max_oom,
        color='C1',
        alpha=0.3,
        edgecolor=None
    )
    ax_gamma.fill_between(
        df_diag_2['t']*1e12,
        df_diag_2['gamma']/10**gamma_plus_sigma_max_oom-\
        df_diag_2['sigma gamma']/10**gamma_plus_sigma_max_oom,
        df_diag_2['gamma']/10**gamma_plus_sigma_max_oom+\
        df_diag_2['sigma gamma']/10**gamma_plus_sigma_max_oom,
        color='C3',
        alpha=0.3,
        edgecolor=None
    )
    
    fig.colorbar(im_lwfa, cax=ax_lwfa_cb, extend='max')
    
    # remove unneccessary tick labels
    ax_q.set_xticklabels([])
    ax_gamma.set_xticklabels([])
    ax_lambda.set_xticklabels([])
    ax_a0.set_xticklabels([])
    
    # change tick positions
    ax_density_profile.xaxis.tick_top()
    ax_density_profile.xaxis.set_label_position('top')
    
    # set precision
    y_major_formatter = FormatStrFormatter('%.2f')
    ax_q.yaxis.set_major_formatter(y_major_formatter)
    ax_lambda.yaxis.set_major_formatter(y_major_formatter)
    ax_w.yaxis.set_major_formatter(y_major_formatter)
    ax_gamma.yaxis.set_major_formatter(y_major_formatter)
    ax_a0.yaxis.set_major_formatter(y_major_formatter)
    ax_ct.yaxis.set_major_formatter(y_major_formatter)

    # set axis labels
    ax_density_profile.set_xlabel('$z$ ($\mathrm{\mu}$m)')
    ax_density_profile.set_ylabel(r'$n_e$ ($\times 10 ^ {{{}}}$'.format(
        n_profile_max_oom-6) + ' cm$^{-3}$)')
    if z_units == "window":
        ax_lwfa.set_xlabel('$\zeta$ ($\mathrm{\mu}$m)')
    else:
        ax_lwfa.set_xlabel('$z$ ($\mathrm{\mu}$m)')
    ax_lwfa.set_ylabel('$r$ ($\mathrm{\mu}$m)')
    ax_lwfa_cb.set_title('$n_e$\n' + r'($\times 10 ^ {{{}}}$ '.format(
        n_e_max_plot_oom-6) + 'cm$^{-3}$)')
    ax_q.set_ylabel(r'$|q_e|$ ($\times 10 ^ {{{}}}$ '.format(q_e_max_oom) + 'C)')
    ax_gamma.set_ylabel(r'$\hat{\gamma}$' + r' ($\times 10 ^ {{{}}}$'.format(
        gamma_max_oom) + ')')
    ax_lambda.set_ylabel(r'$\lambda_\mathrm{peak}$ ' + \
                        r'($\times 10 ^ {{{}}}$ '.format(lambda_max_oom) + 'm)')
    ax_a0.set_ylabel(r'$a_0$ ($\times 10 ^ {{{}}}$'.format(a0_max_oom) + ')')
    ax_w.set_ylabel(r'$w_l$ ($\times 10 ^ {{{}}}$ '.format(w_max_oom) + 'm)')
    ax_w.set_xlabel(r'$t$ (ps)')
    ax_ct.set_ylabel(r'$c \tau_l$ ($\times 10 ^ {{{}}}$ '.format(ct_max_oom) +\
        'm)')
    ax_ct.set_xlabel(r'$t$ (ps)')

    fig.align_ylabels([ax_q, ax_lambda, ax_w])
    fig.align_ylabels([ax_gamma, ax_a0, ax_ct])

    t_text = fig.text(
        ax_q.get_position().x0+(ax_gamma.get_position().x1-ax_q.get_position().x0)/2,
        ax_q.get_position().y1+(1-ax_q.get_position().y1)/2,
        r'$t = {:.1f}$ ps'.format(ts_1.t[0]*1e12),
        va='center', ha='center', fontsize=11
    )
    
    fig.canvas.draw()
    ax_density_profile.set_ylim(0, ax_density_profile.get_ylim()[1])
    ax_density_profile.fill_between(
        [info_rho_1.z[0]*1e6, info_rho_1.z[-1]*1e6],
        [0, 0],
        [ax_density_profile.get_ylim()[1], ax_density_profile.get_ylim()[1]],
        color='C1',
        alpha=0.5,
        edgecolor=None
    )
    
    def animate_plasma_density_and_info_compare(i):
        print_string = str(i+1) + ' / ' + str(
            min([ts_1.iterations.size, ts_2.iterations.size])
        )
        print(print_string.ljust(20), end="\r", flush=True)
        
        # get data
        rho_1, info_rho_1 = ts_1.get_field(
            iteration=ts_1.iterations[i], field='rho'
        )
        rho_2, info_rho_2 = ts_2.get_field(
            iteration=ts_2.iterations[i], field='rho'
        )
        n_e_1 = np.abs(rho_1) / e
        n_e_2 = np.abs(rho_2) / e
        env_1, info_env_1 = lpd_1.get_laser_envelope(
            iteration=ts_1.iterations[i], pol=pol_1
        )
        env_2, info_env_2 = lpd_2.get_laser_envelope(
            iteration=ts_2.iterations[i], pol=pol_2
        )
        
        # redraw the data
        n_e_im = _glue_images(
            np.divide(n_e_1[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
                    10.0**n_e_max_plot_oom, dtype=np.float64),
            np.divide(n_e_2[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
                    10.0**n_e_max_plot_oom, dtype=np.float64)
        )
        
        im_lwfa.set_data(n_e_im)
        
        ax_q_scat_1.set_data(
            df_diag_1['t'][i]*1e12,
            np.abs(df_diag_1['q'][i])/10**q_e_max_oom
        )
        ax_q_scat_2.set_data(
            df_diag_2['t'][i]*1e12,
            np.abs(df_diag_2['q'][i])/10**q_e_max_oom
        )
        ax_q_line_1.set_data(
            df_diag_1['t'][:i+1]*1e12,
            np.abs(df_diag_1['q'][:i+1])/10**q_e_max_oom
        )
        ax_q_line_2.set_data(
            df_diag_2['t'][:i+1]*1e12,
            np.abs(df_diag_2['q'][:i+1])/10**q_e_max_oom
        )
        
        ax_gamma_scat_1.set_data(
            df_diag_1['t'][i]*1e12,
            df_diag_1['gamma'][i]/10**gamma_plus_sigma_max_oom
        )
        ax_gamma_scat_2.set_data(
            df_diag_2['t'][i]*1e12,
            df_diag_2['gamma'][i]/10**gamma_plus_sigma_max_oom
        )
        ax_gamma_line_1.set_data(
            df_diag_1['t'][:i+1]*1e12,
            df_diag_1['gamma'][:i+1]/10**gamma_plus_sigma_max_oom
        )
        ax_gamma_line_2.set_data(
            df_diag_2['t'][:i+1]*1e12,
            df_diag_2['gamma'][:i+1]/10**gamma_plus_sigma_max_oom
        )
        
        ax_lambda_scat_1.set_data(
            df_diag_1['t'][i]*1e12,
            df_diag_1['lambda'][i]/10**lambda_max_oom
        )
        ax_lambda_scat_2.set_data(
            df_diag_2['t'][i]*1e12,
            df_diag_2['lambda'][i]/10**lambda_max_oom
        )
        ax_lambda_line_1.set_data(
            df_diag_1['t'][:i+1]*1e12,
            df_diag_1['lambda'][:i+1]/10**lambda_max_oom
        )
        ax_lambda_line_2.set_data(
            df_diag_2['t'][:i+1]*1e12,
            df_diag_2['lambda'][:i+1]/10**lambda_max_oom
        )
        
        ax_a0_scat_1.set_data(
            df_diag_1['t'][i]*1e12,
            df_diag_1['a0'][i]/10**a0_max_oom
        )
        ax_a0_scat_2.set_data(
            df_diag_2['t'][i]*1e12,
            df_diag_2['a0'][i]/10**a0_max_oom
        )
        ax_a0_line_1.set_data(
            df_diag_1['t'][:i+1]*1e12,
            df_diag_1['a0'][:i+1]/10**a0_max_oom
        )
        ax_a0_line_2.set_data(
            df_diag_2['t'][:i+1]*1e12,
            df_diag_2['a0'][:i+1]/10**a0_max_oom
        )
        
        ax_w_scat_1.set_data(
            df_diag_1['t'][i]*1e12,
            df_diag_1['w'][i]/10**w_max_oom
        )
        ax_w_scat_2.set_data(
            df_diag_2['t'][i]*1e12,
            df_diag_2['w'][i]/10**w_max_oom
        )
        ax_w_line_1.set_data(
            df_diag_1['t'][:i+1]*1e12,
            df_diag_1['w'][:i+1]/10**w_max_oom
        )
        ax_w_line_2.set_data(
            df_diag_2['t'][:i+1]*1e12,
            df_diag_2['w'][:i+1]/10**w_max_oom
        )
        
        ax_ct_scat_1.set_data(
            df_diag_1['t'][i]*1e12,
            df_diag_1['ct'][i]/10**ct_max_oom
        )
        ax_ct_scat_2.set_data(
            df_diag_2['t'][i]*1e12,
            df_diag_2['ct'][i]/10**ct_max_oom
        )
        ax_ct_line_1.set_data(
            df_diag_1['t'][:i+1]*1e12,
            df_diag_1['ct'][:i+1]/10**ct_max_oom
        )
        ax_ct_line_2.set_data(
            df_diag_2['t'][:i+1]*1e12,
            df_diag_2['ct'][:i+1]/10**ct_max_oom
        )
        
        if z_units == "simulation":
            extent[0] = info_rho_1.z[z_low_idx]
            if z_up_idx is None:
                extent[1] = info_rho_1.z[-1]
            else:
                extent[1] = info_rho_1.z[z_up_idx]
            im_lwfa.set_extent(extent*1e6)
        
        t_text.set_text(r'$t = {:.1f}$ ps'.format(ts_1.t[i]*1e12))
        
        ax_density_profile.collections.clear()
        ax_density_profile.fill_between(
            [info_rho_1.z[0]*1e6, info_rho_1.z[-1]*1e6],     
            [0, 0],
            [ax_density_profile.get_ylim()[1],
             ax_density_profile.get_ylim()[1]],
            color='C1',
            alpha=0.5,
            edgecolor=None
        )
        
        ax_gamma.collections.clear()
        if i > 0:
            ax_gamma.fill_between(
                df_diag_1['t'][:i+1]*1e12,
                df_diag_1['gamma'][:i+1]/10**gamma_plus_sigma_max_oom-\
                df_diag_1['sigma gamma'][:i+1]/10**gamma_plus_sigma_max_oom,
                df_diag_1['gamma'][:i+1]/10**gamma_plus_sigma_max_oom+\
                df_diag_1['sigma gamma'][:i+1]/10**gamma_plus_sigma_max_oom,
                color='C1',
                alpha=0.3,
                edgecolor=None
            )
            ax_gamma.fill_between(
                df_diag_2['t'][:i+1]*1e12,
                df_diag_2['gamma'][:i+1]/10**gamma_plus_sigma_max_oom-\
                df_diag_2['sigma gamma'][:i+1]/10**gamma_plus_sigma_max_oom,
                df_diag_2['gamma'][:i+1]/10**gamma_plus_sigma_max_oom+\
                df_diag_2['sigma gamma'][:i+1]/10**gamma_plus_sigma_max_oom,
                color='C3',
                alpha=0.3,
                edgecolor=None
            )
        
        if z_units == "window":
            cs_z = info_rho_1.z[z_low_idx:z_up_idx] - info_rho_1.z[z_low_idx]
        else:
            cs_z = info_rho_1.z[z_low_idx:z_up_idx]
        for c in cs_lwfa[0].collections:
            c.remove()
        env_im = _glue_images(
            env_1[r_low_idx:r_up_idx, z_low_idx:z_up_idx],
            env_2[r_low_idx:r_up_idx, z_low_idx:z_up_idx]
        )
        cs_lwfa[0] = ax_lwfa.contour(
            cs_z*1e6,
            info_env_1.r[r_low_idx:r_up_idx]*1e6,
            env_im,
            levels=env_im.max()*np.linspace(1/np.e**2, 0.95, 4),
            cmap=laser_cmap
        )
        im_lwfa.set_extent(extent*1e6)
        
        div_z = (ax_lwfa.get_xlim())
        div_line.set_data(div_z, [0, 0])
        
        return cs_lwfa
    
    print("Animating")
    ani_range = min([ts_1.iterations.size, ts_2.iterations.size])
    ani = animation.FuncAnimation(
        fig, animate_plasma_density_and_info_compare,
        range(0, ani_range), repeat=False, blit=False,
        interval=interval
    )
    
    ani.save(os.path.join(out_dir, "complete_compare.mp4"))
    
    
    
    
def _load_number_density(n_file, z_min=None, z_max=None, z_points=None):
    '''Load a number density profile.
    
    Can either be a `.py` or `.csv` file.
    
    If `.py`, the file should contain a function called `number_density`. The
    function should take one argument: a value of z-position, and should return
    a corresponding number density value. See `plasma_profile_example.py` for
    an example.
    
    If `.csv`, the file should be a comma-separated list of coordinate-number
    density value pairs. The first line of the csv file is ignored.
    
    Parameters
    ----------
    n_file : str
        Path to the `.py` or `.csv` file.
    z_min : float, optional
        Minimum value of z-coordinate. Required if using `.py` file. Not used
        if using `.csv` file.
    z_max : float, optional
        Maximum value of z-coordinate. Required if using `.py` file. Not used
        if using `.csv` file.
    z_points : int, optional
        Number of points to sample the profile. Required if using `.py` file.
        Not used if using `.csv` file.
        
    Returns
    -------
    z_array : ndarray
        List of coordinate values.
    n_array : ndarray
        List of number density values.
    '''
    if n_file.endswith('.py'):
        if None in [z_min, z_max, z_points]:
            err_str = "One of z_min, z_max, z_points not defined but all " + \
                      "are required when generating plasma profile from py file"
            raise ValueError(err_str)
        z_array = np.linspace(z_min, z_max, z_points)
        spec = importlib.util.spec_from_file_location(
            "plasma_profile",
            n_file
        )
        number_density_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(number_density_mod)
        n_array = np.zeros((z_points,))
        print("Generating number density profile")
        for i in tqdm(range(len(z_array))):
            n_array[i] = number_density_mod.number_density(z_array[i])
    else:
        z_array, n_array = [], []
        with open(n_file) as c:
            c_reader = csv.reader(c, delimiter=',')
            lc = 0
            for row in c_reader:
                if lc != 0:
                    z_array.append(float(row[0]))
                    n_array.append(float(row[1]))
                lc += 1
        
    return np.array(z_array), np.array(n_array)



def _glue_images(im1, im2):
    '''Glue the top half if im1 and the bottom half of im2 together.'''
    im_glued = np.concatenate(
        (
            im1[:im1.shape[0]//2, :],
            im2[im2.shape[0]//2:, :]
        ),
        axis=0
    )
    return im_glued
    
    


# unit testing
if __name__ == "__main__":
    
    # animated_plasma_density(
    #     sys.argv[1],
    #     z_units='simulation',
    #     n_max=1e24,
    #     # relative_n_max=0.003
    # )
    
    # animated_plasma_density_and_info(
    #     sys.argv[1],
    #     sys.argv[2],
    #     -100e-6,
    #     0.0023,
    #     # n_max=1e24,
    #     relative_n_max=0.003
    # )
    
    animated_plasma_density_and_info_compare(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        -100e-6,
        2300e-6,
        sys.argv[5],
        n_max=3e24,
        z_points=100,
        r_roi=40e-6,
    )
    
    # lpd = LpaDiagnostics(os.path.join(sys.argv[1], "diags", "hdf5"))
    # env, _ = lpd.get_laser_envelope(iteration=0, pol='y')
    # plt.figure()
    # plt.imshow(env)
    # plt.show()