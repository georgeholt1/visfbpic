# Author: George K. Holt
# License: MIT
# Version: 0.0.1
"""
Part of VISFBPIC.

Contains methods for creating animations of the FBPIC data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
from scipy.constants import e, c, pi
from tqdm import tqdm
import sys
import importlib.util

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
    relative_n_max=0.003,
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
    relative_n_max : float, optional
        The maximum value of the colour scale relative to the maximum number
        density at any time in the simulation. Defaults to 0.003.
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
    n_e_max_plot = n_e_max * relative_n_max
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
        r_low_idx = _find_nearest_idx(info_rho.r, -r_roi)
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
        ax.set_xlabel('$\zeta$ $\mathrm{\mu}$m)')
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
    relative_n_max=0.003,
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
        Path to file containing a function called `number_density`. The function
        should take one argument: a value of z-position, and should return a
        corresponding number density value. See `plasma_profile_example.py` for
        an example.
    z_min, zmax : float
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
    relative_n_max : float, optional
        The maximum value of the colour scale relative to the maximum number
        density at any time in the simulation. Defaults to 0.003.
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
    
    # generate number density distribution
    spec = importlib.util.spec_from_file_location(
        "plasma_profile",
        n_file
    )
    number_density_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(number_density_mod)
    z_array = np.linspace(z_min, z_max, z_points)
    n_array = []
    for z in z_array:
        n_array.append(number_density_mod.number_density(z))
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
    try:
        gamma_max_oom = int(np.floor(np.log10(gamma_max)))
    except OverflowError:
        gamma_max_oom = 0
    try:
        gamma_plus_sigma_max_oom = int(np.floor(np.log10(gamma_plus_sigma_max)))
    except OverflowError:
        gamma_plus_sigma_max_oom = 0
    lambda_max_oom = int(np.floor(np.log10(lambda_max)))
    a0_max_oom = int(np.floor(np.log10(a0_max)))
    w_max_oom = int(np.floor(np.log10(w_max)))
    ct_max_oom = int(np.floor(np.log10(ct_max)))
    
    # get maximum number density for plot
    n_e_max_plot = n_e_max * relative_n_max
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
        r_low_idx = _find_nearest_idx(info_rho.r, -r_roi)
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
        wspace=0.41
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

    # set axis labels
    ax_density_profile.set_xlabel('$z$ ($\mathrm{\mu}$m)')
    ax_density_profile.set_ylabel(r'$n_e$ ($\times 10 ^ {{{}}}$'.format(
        n_profile_max_oom-6) + ' cm$^{-3}$)')
    if z_units == "window":
        ax_lwfa.set_xlabel('$\zeta$ ($\mathrm{\mu}$m)')
    else:
        ax_lwfa.set_xlabel('$z$ ($\mathrm{\mu}$m)')
    ax_lwfa.set_ylabel('$r$ ($\mathrm{\mu}$m)')
    ax_lwfa_cb.set_title(r'$n_e$ ($\times 10 ^ {{{}}}$ '.format(
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


# unit testing
if __name__ == "__main__":
    # animated_plasma_density(sys.argv[1], z_units='simulation')
    animated_plasma_density_and_info(
        sys.argv[1],
        sys.argv[2],
        -125e-6,
        0.0083
    )