from typing import List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from string import ascii_lowercase
from math import floor
import xarray as xr
import seaborn

from helper_funcs import convert2rgb, get_tab_colors
import data_tools as dt

def plot_metric(data_paths:List[str], metric:str,
    lower_bound:Optional[str] = None, upper_bound:Optional[str] = None, alpha = 0.5,
    normalize:bool = False, labels:Optional[List[str]] = None, ylim = None, xlim = None,
    title:Optional[str] = None, logscale:Optional[str] = False, ylabel = None):
    """Plots a metric evolution from the given data_paths.

    Since metrics require an evaluation of the forward model,
    the last iteration must be discarded (only parameters available).

    Args:
     - data_paths: List of diagnostic data paths from which to draw
      the metrics.
     - metric: Name of the metric to be plotted.
     - lower_bound: If given, shades the area between the metric
      and this lower bound metric from the same dataset.
     - upper_bound: If given, shades the area between the metric
      and this upper bound metric from the same dataset.
     - normalize: If True, normalizes the metric with respect to
      the metric at t=0.
    """
    tab_colors = get_tab_colors()
    fig = plt.figure(metric)
    if title:
        plt.title(title)
    max_t = 0
    for (i, data_path) in enumerate(data_paths):
        shading = convert2rgb(tab_colors[i], 0.4)
        var = dt.ncFetch(data_path, 'metrics', metric)[:-1]
        if normalize:
            den = var[0]
            var = var/den
        else:
            den = 1.
        if labels is not None:
            label = labels[i]
        else:
            label = (data_path).split('_')[-1]
        t = dt.ncFetch(data_path, 'metrics', 'iteration')[:-1]
        max_t = max(max_t, t[-1])
        plt.plot(t, var, color=tab_colors[i], label = label)
        if lower_bound is not None:
            low = dt.ncFetch(data_path, 'metrics', lower_bound)[:-1]/den
            plt.fill_between(t, var, low, color=shading, alpha = alpha)
        if upper_bound is not None:
            upp = dt.ncFetch(data_path, 'metrics', upper_bound)[:-1]/den
            plt.fill_between(t, var, upp, color=shading, alpha = alpha)
    if logscale:
        plt.yscale("log")
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(metric)
    plt.xlabel('Iteration')
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim(0, max_t)
    plt.ylim(ylim)
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(frameon=True, prop={'size': 7}, loc = 'upper right')
    plt.tight_layout()
    plt.savefig(metric+'.pdf', format='pdf')
    return



def plot_var_mse(ds_dict:dict,
                nrows:int = 3,
                ncols:int = 2,
                save_fig_path:str = False):


    var_names = ds_dict["reference"]["ref_variable_names"].isel(config = 0).values

    fig, axs = plt.subplots(nrows, ncols, sharex = True, figsize=(ncols * 7, nrows * 4))
    if nrows > 1 and ncols > 1:
        axs = axs.flatten()

    for var_name_i, ax in zip(range(len(var_names)), axs):
        var_name = var_names[var_name_i]
        mse_field_name = "mse_{}_full".format(var_name)

        u_dataframe = ds_dict["particle_diags"][mse_field_name].to_dataframe().reset_index()

        seaborn.stripplot(x = u_dataframe['iteration'],
                             y = u_dataframe[mse_field_name],
                             data = u_dataframe[['iteration', mse_field_name]],
                             ax = ax,
                             c = 'k',
                             alpha = 0.3,
                             size = 2.0)

        u_dataframe["iteration"] = u_dataframe["iteration"] - 1
        mean_values = u_dataframe.groupby('iteration')[mse_field_name].mean()
        ax.plot(mean_values, label='Mean', color='blue')
        ax.set_title(var_name, weight = "bold")

        max_val = u_dataframe[mse_field_name].max()
        ax.set_ylim(0, max_val)

        ax.set_ylabel('')
        ax.grid()

    if save_fig_path:
        dir_name = os.path.dirname(save_fig_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_fig_path, dpi=200)

def construct_profiles(ds_dict:dict, iteration:int = 1, integrated_quantities:list = ['lwp_mean']):
    """
    Unpack profiles (and integrated quanities like LWP) from g_full and y_full vectors.
    """

    ds_ref = ds_dict["reference"]
    ds_particle = ds_dict["particle_diags"]
    var_names = ds_ref["ref_variable_names"].isel(config = 0).values
    num_vars = len(var_names)
    num_cases = len(ds_ref['config'])

    num_particles = len(ds_particle.particle)
    dof = len(ds_ref.dof)
    norm_factor = ds_ref.norm_factor

    all_g = np.zeros((num_cases, num_vars, num_particles, dof)) #(num_cases, num_vars, num_particles, num_grid_points)
    all_g_integrated = np.zeros((num_cases, num_vars, num_particles, 1))
    all_y = np.zeros((num_cases, num_vars, dof))
    all_y_integrated = np.zeros((num_cases, num_vars, 1))

    print("Constructing profiles for iteration ", iteration)

    idx_y_full = 0
    for case_num in range(num_cases):
        for var_ind in range(ds_ref.num_vars[case_num].item()):
            var_name = var_names[var_ind]
            if var_name in integrated_quantities:
                dof = 1
            else:
                dof = np.count_nonzero(ds_ref["config_z_obs"].isel(config = case_num))

            y_full_train = ds_ref.y_full.values[idx_y_full:idx_y_full+int(dof)]*np.sqrt(norm_factor[var_ind, case_num]).item()
            if dof != 1:
                all_y[case_num, var_ind, :dof] = y_full_train
            elif dof == 1:
                all_y_integrated[case_num, var_ind, :] = y_full_train
            for particle_num in range(num_particles):
                g_y_val = ds_particle.g_full.isel(particle = particle_num, iteration = iteration).values[idx_y_full:idx_y_full+int(dof)]*np.sqrt(norm_factor[var_ind, case_num]).item()
                if dof != 1:
                    all_g[case_num, var_ind, particle_num, :dof] = g_y_val
                elif dof == 1:
                    all_g_integrated[case_num, var_ind, particle_num, :] = g_y_val

            idx_y_full += int(dof)

    out_dict = {}
    out_dict["all_g"] = all_g
    out_dict["all_g_integrated"] = all_g_integrated
    out_dict["all_y"] = all_y
    out_dict["all_y_integrated"] = all_y_integrated
    return out_dict

def plot_y_profiles(ds_dict:dict,
                    y_dict:dict,
                    case_i:int = 0,
                    nrows:int = 2,
                    ncols:int = 3,
                    integrated_quantities:list = ['lwp_mean'],
                    save_fig_path:str = False):

    """Plot profiles (and integrated quanities like LWP) from dict created from `construct_profiles`"""

    ds_ref = ds_dict["reference"]
    var_names = ds_ref["ref_variable_names"].isel(config = 0).values

    num_vars = len(var_names)
    z_val = ds_ref.config_z_obs.values[:,case_i]

    fig , ax = plt.subplots(nrows, ncols, sharey = True, figsize = (15,8))
    for var_i in range(y_dict["all_g"].shape[1]):
        ax_i = ax[np.unravel_index(var_i, (nrows, ncols))]
        var_name = var_names[var_i]
        z_case_i = ds_ref["config_z_obs"].isel(config = case_i)
        if var_name in integrated_quantities:
            ax_i.axvline(x=y_dict["all_y_integrated"][case_i, var_i,:].item(), color='k', linestyle='-')
            x_var = y_dict["all_g_integrated"][case_i, var_i,:]
            ax_i.scatter(x_var, (z_case_i.max().item()/2)*np.ones(len(x_var)), s = 10, alpha = 0.3)
        else:
            dof = np.count_nonzero(z_case_i)
            ax_i.plot(y_dict["all_g"][case_i, var_i,:,:dof].T, z_val[:dof], alpha = 0.2)
            ax_i.plot(y_dict["all_y"][case_i, var_i, :dof], z_val[:dof], c = 'k')

        ax_i.set_title(var_names[var_i])
        ax_i.set_ylim(0, z_case_i.max())
        ax_i.grid()
    plt.suptitle(ds_ref['config_name'].values[case_i])

    if save_fig_path:
        dir_name = os.path.dirname(save_fig_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_fig_path, dpi=200)


def plot_y_profiles_iter(ds_dict_full, profiles, iterations:list = [1, 5], save_fig_dir = "./figs/y_profiles"):
    case_names = ds_dict_full["reference"]["config_name"].values
    for iteration in iterations:
        profiles = construct_profiles(ds_dict_full, iteration = iteration)
        for case_i in range(len(case_names)):
            case_name_i = case_names[case_i]
            save_fig_path = os.path.join(save_fig_dir, "iter_{}/case{}.pdf".format(iteration, case_i))
            plot_y_profiles(ds_dict_full, profiles, case_i = case_i, save_fig_path = save_fig_path)



def plot_epoch_avg_metric(data_paths:List[str], metric:str,
    lower_bound:Optional[str] = None, upper_bound:Optional[str] = None,
    normalize:bool = False, labels:Optional[List[str]] = None):
    """Plots an epoch-averaged metric evolution from the given data_paths.

    Since metrics require an evaluation of the forward model,
    the last iteration must be discarded (only parameters available).

    Args:
     - data_paths: List of diagnostic data paths from which to draw
      the metrics.
     - metric: Name of the metric to be plotted.
     - lower_bound: If given, shades the area between the metric
      and this lower bound metric from the same dataset.
     - upper_bound: If given, shades the area between the metric
      and this upper bound metric from the same dataset.
     - normalize: If True, normalizes the metric with respect to
      the metric at t=0.
    """
    tab_colors = get_tab_colors()
    fig = plt.figure(metric+' per epoch')
    for (i, data_path) in enumerate(data_paths):
        shading = convert2rgb(tab_colors[i], 0.4)
        batch_size = dt.ncFetchDim(data_path, 'reference', 'batch_size')
        config_num = dt.ncFetchDim(data_path, 'reference', 'config')
        iter_per_epoch = int(config_num/batch_size) # Should be read from file
        var = dt.ncFetch(data_path, 'metrics', metric)[:-1]
        var = iter_to_epochs(var, iter_per_epoch)
        epoch = np.arange(len(var))
        if normalize:
            den = var[0]
            var = var/den
        else:
            den = 1.
        if labels is not None:
            label = labels[i]
        else:
            label = (data_path).split('_')[-1]
        plt.plot(epoch, var, color=tab_colors[i], label = label)
        if lower_bound is not None:
            low = dt.ncFetch(data_path, 'metrics', lower_bound)[:-1]
            low = iter_to_epochs(low, iter_per_epoch)/den
            plt.fill_between(epoch, var, low, color=shading)
        if upper_bound is not None:
            upp = dt.ncFetch(data_path, 'metrics', upper_bound)[:-1]
            upp = iter_to_epochs(upp, iter_per_epoch)/den
            plt.fill_between(epoch, var, upp, color=shading)
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        ax = fig.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(frameon=False)
        plt.savefig(metric+'_per_epoch.pdf', format='pdf')
    return

def iter_to_epochs(var:np.ndarray, iter_per_epoch:int) -> np.ndarray:
    """Converts an iteration-dependent array to an epoch-dependent array through averaging.

    The operation ignores any remainder iterations of unfinished training epochs.

    Args:
     - var: Array with iterations as a dimension.
     - iter_per_epoch: Number of iterations per epoch.
    """
    # Set epochs as rows
    var = np.reshape(
        var[:len(var) - len(var)%iter_per_epoch], (-1, iter_per_epoch))
    # Average metric per epoch
    var = np.mean(var, axis=1)
    return var

def plot_y_full(data_path:str, var_names:Optional[List[str]] = None):
    """Plots vertical profiles from `y_full` for each variable and
    configuration.

    Args:
     - data_path: Diagnostic data path from which to draw
      the metrics.
     - var_names: List of variable names for each configuration,
      assuming they are constant across configurations.
    """
    config_names = dt.ncFetch(data_path, 'reference', 'config_name')
    config_z = dt.ncFetch(data_path, 'reference', 'config_z_obs_val')
    y_full = dt.ncFetch(data_path, 'reference', 'y_full')
    var_dof = dt.ncFetch(data_path, 'reference', 'var_dof')
    num_vars = dt.ncFetch(data_path, 'reference', 'num_vars')
    tab_colors = get_tab_colors()
    # Loop over configs
    idx_y_full = 0
    for (c, (num_var, dof, config_name)) in enumerate(
            zip(num_vars, var_dof, config_names)):
        for v in range(int(num_var)):
            if var_names is not None:
                var_name = var_names[v]
            else:
                var_name = str(v)
            title = 'y_full_'+config_name+'_var_'+var_name
            plt.figure(title)
            profile = y_full[idx_y_full:idx_y_full+int(dof)]
            z = config_z
            plt.plot(profile, z, color=tab_colors[c])
            # Decorate plot
            plt.ylabel(r'$z$ (m)')
            plt.xlabel(var_name)
            delta_profile = max(profile) - min(profile)
            plt.xlim(min(profile) - 0.1*delta_profile,
                max(profile) + 0.1*delta_profile)
            plt.ylim(0, 3000)
            plt.tight_layout()
            # Save plot and update index
            plt.savefig(title+'.pdf', format='pdf')
            idx_y_full += int(dof)
    return

def plot_prior_post_ref(data_path:str, var_names:Optional[List[str]] = None, validation:bool =True):
    """Plots vertical profiles of the reference/truth, best initial particle and
    best final particle for each variable and configuration.

    Args:
     - data_path: Diagnostic data path from which to draw
      the metrics.
     - var_names: List of variable names for each configuration,
      assuming they are constant across configurations.
    """
    if validation:
        num_vars = dt.ncFetch(data_path, 'reference', 'num_vars_val')
        var_dof = dt.ncFetch(data_path, 'reference', 'var_dof_val')
        y_full = dt.ncFetch(data_path, 'reference', 'y_full_val')
        config_z = dt.ncFetch(data_path, 'reference', 'config_z_obs_val')
        config_names = dt.ncFetch(data_path, 'reference', 'config_name_val')
        norm_factor = dt.ncFetch(data_path, 'reference', 'norm_factor_val')
        # Last forward model evals are empty
        g_full = dt.ncFetch(data_path, 'particle_diags', 'val_g_full')[:-1, :, :]
        mse_full = dt.ncFetch(data_path, 'particle_diags', 'val_mse_full')[:-1, :]
    else:
        num_vars = dt.ncFetch(data_path, 'reference', 'num_vars')
        var_dof = dt.ncFetch(data_path, 'reference', 'var_dof')
        y_full = dt.ncFetch(data_path, 'reference', 'y_full')
        config_z = dt.ncFetch(data_path, 'reference', 'config_z_obs')
        config_names = dt.ncFetch(data_path, 'reference', 'config_name')
        norm_factor = dt.ncFetch(data_path, 'reference', 'norm_factor')
        # Last forward model evals are empty
        g_full = dt.ncFetch(data_path, 'particle_diags', 'g_full')[:-1, :, :]
        mse_full = dt.ncFetch(data_path, 'particle_diags', 'mse_full')[:-1, :]

    print("Shape of mse_full is", np.shape(mse_full))
    print("Shape of g_full is", np.shape(g_full))
    best_particle = np.argmin(mse_full, axis = 1)
    print("Shape of best_particle is", np.shape(best_particle))
    tab_colors = get_tab_colors()
    # Loop over configs
    idx_y_full = 0
    for (c, (num_var, dof, config_name)) in enumerate(
            zip(num_vars, var_dof, config_names)):
        for v in range(int(num_var) - 1):
            if var_names is not None:
                var_name = var_names[v]
            else:
                var_name = str(v)
            title = 'y_full_'+config_name+'_var_'+var_name
            plt.figure(title)
            ref_profile = y_full[idx_y_full:idx_y_full+int(dof)]*np.sqrt(norm_factor[v, c])
            prior_profile = g_full[0, idx_y_full:idx_y_full+int(dof), best_particle[0]]*np.sqrt(norm_factor[v, c])
            post_profile = g_full[-1, idx_y_full:idx_y_full+int(dof), best_particle[-1]]*np.sqrt(norm_factor[v, c])
            z = config_z[:, c]
            plt.plot(ref_profile, z, color='0.2', label='LES')
            plt.plot(prior_profile, z, color='tab:orange', label='Prior')
            plt.plot(post_profile, z, color='tab:green', label='Posterior')
            # Decorate plot
            plt.ylabel(r'$z$ (m)')
            plt.xlabel(var_name)
            delta_profile = max(ref_profile) - min(ref_profile)
            plt.xlim(min(ref_profile) - 0.1*delta_profile,
                max(ref_profile) + 0.1*delta_profile)
            plt.ylim(0, z.max())
            plt.legend(frameon=False)
            plt.ticklabel_format(axis="x", style="sci", scilimits=(-2, 4))
            plt.tight_layout()
            # Save plot and update index
            plt.savefig(title+'.pdf', format='pdf')
            idx_y_full += int(dof)
    return

def plot_cov_spectrum(data_paths:List[str]):
    """Plots vertical profiles from `y_full` for each variable and
    configuration.

    Args:
     - data_path: Diagnostic data path from which to draw
      the metrics.
     - var_names: List of variable names for each configuration,
      assuming they are constant across configurations.
    """
    tab_colors = get_tab_colors()
    for data_path in data_paths:
        gamma_full = dt.ncFetch(data_path, 'reference', 'Gamma_full')
        gamma = dt.ncFetch(data_path, 'reference', 'Gamma')
        eig_full, _ = np.linalg.eig(gamma_full)
        eig_full = sorted(eig_full, reverse=True)
        eig = sorted(np.diagonal(gamma), reverse=True)
        title = 'spectrum_'+os.path.basename(data_path)
        plt.figure(title)
        plt.plot(range(len(eig_full)), eig_full, color='tab:orange')
        plt.plot(range(len(eig)), eig, color='tab:green')
        plt.yscale('log')
        # Decorate plot
        plt.ylabel(r'Eigenvalue')
        plt.xlabel(r'Mode')
        plt.ylim(min(eig_full), 1.05*max(eig))
        plt.xlim(0, len(eig_full))
        plt.tight_layout()
        # Save plot and update index
        plt.savefig(title+'.pdf', format='pdf')
    return

def plot_modes(data_path:str, num_modes:int = 5, same_plot:bool = False):
    """Plots leading `num_modes` of each ReferenceModel in the data_path
    diagnostics.

    Args:
     - data_path: Diagnostic data path from which to draw the modes.
     - num_modes: Modes to plot from each ReferenceModel.
     - same_plot: Whether to plot modes in the same figure or different ones.
    """
    config_names = dt.ncFetch(data_path, 'reference', 'config_name')
    d_csum = np.cumsum(dt.ncFetch(data_path, 'reference', 'config_pca_dim'))
    d_csum = np.insert(d_csum, 0, 0)
    var_dof = dt.ncFetch(data_path, 'reference', 'var_dof')
    num_vars = dt.ncFetch(data_path, 'reference', 'num_vars')
    config_dofsum = np.cumsum(np.multiply(num_vars, var_dof))
    config_dofsum = np.insert(config_dofsum, 0, 0)
    P_pca = dt.ncFetch(data_path, 'reference', 'P_pca')
    tab_colors = get_tab_colors()
    # Loop over configs
    for (c, (config_dofs, d_cs, config_name)) in enumerate(
            zip(config_dofsum, d_csum, config_names)):
        # PCA matrix for this configuration
        modes = P_pca[d_cs:d_csum[c+1], config_dofs:config_dofsum[c+1]]
        # Number of principal modes
        max_modes = d_csum[c+1] - d_cs
        if same_plot:
            title = 'modes_'+config_name
            plt.figure(title)
        for m in range(min(num_modes, max_modes)):
            if not same_plot:
                title = 'mode_'+str(m)+'_'+config_name
                plt.figure(title)
            # Leading eigenvectors are last in julia eig(), so reverse order
            mode = modes[max_modes - 1 - m, :]
            plt.plot(range(len(mode)), mode, color=tab_colors[m], label='Mode'+str(m))
            # Decorate plot
            plt.ylabel(r'Magnitude')
            plt.xlabel(r'Degree of Freedom')
            plt.legend(frameon=False)
            plt.tight_layout()
            # Save plot and update index
            if not same_plot:
                plt.savefig(title+'.pdf', format='pdf')
        if same_plot:
            plt.savefig(title+'.pdf', format='pdf')
    return

def plot_eigval_zero_crossings(data_path:str, normalize:bool = False):
    """Plots scatterplot of eigenvalue magnitude with the number of
    zero-crossings of the corresponding eigenmode, for each ReferenceModel.

    Args:
     - data_path: Diagnostic data path from which to draw
      the metrics.
     - normalize: Whether to normalize eigenvalues by the maximum
        eigenvalue in the ReferenceModel.
    """
    config_names = dt.ncFetch(data_path, 'reference', 'config_name')
    d_csum = np.cumsum(dt.ncFetch(data_path, 'reference', 'config_pca_dim'))
    d_csum = np.insert(d_csum, 0, 0)
    gamma = dt.ncFetch(data_path, 'reference', 'Gamma')
    var_dof = dt.ncFetch(data_path, 'reference', 'var_dof')
    num_vars = dt.ncFetch(data_path, 'reference', 'num_vars')
    config_dofsum = np.cumsum(np.multiply(num_vars, var_dof))
    config_dofsum = np.insert(config_dofsum, 0, 0)
    P_pca = dt.ncFetch(data_path, 'reference', 'P_pca')
    tab_colors = get_tab_colors()
    title = 'zero_crossings'
    if normalize:
        title = title+'_normalized'
    plt.figure(title)
    # Loop over configs
    for (c, (config_dofs, d_cs, config_name)) in enumerate(
            zip(config_dofsum, d_csum, config_names)):
        # PCA matrix for this configuration
        modes = P_pca[d_cs:d_csum[c+1], config_dofs:config_dofsum[c+1]]
        # Number of principal modes
        max_modes = d_csum[c+1] - d_cs
        zero_crossing_num = np.zeros(max_modes)
        for m in range(max_modes):
            # Leading eigenvectors are last in julia eig(), so reverse order
            mode = modes[max_modes - 1 - m, :]
            zero_crossings = np.where(np.diff(np.sign(mode)))[0]
            zero_crossing_num[m] = len(zero_crossings)
        # Get eigenvalues and order them
        eigvals_ = np.array(sorted(np.diagonal(gamma[d_cs:d_csum[c+1], d_cs:d_csum[c+1]]),
            reverse=True))
        if normalize:
            den = eigvals_[0]
        else:
            den = 1.0
        plt.plot(zero_crossing_num, eigvals_/den, color=tab_colors[c], label=config_name)
        plt.yscale('log')
        # Decorate plot
        plt.xlabel(r'Zero crossings')
        plt.ylabel(r'Eigenvalue')
        plt.legend(frameon=False)
        plt.tight_layout()
        # Save plot and update index
        plt.savefig(title+'.pdf', format='pdf')
    return

def plot_parameter_evol(data_paths:List[str]):
    """Plots the parameter evolution from the given data_paths.

    The plotted results are the parameter means in physical (constrained) space,
    with shading bounds defined by the marginal standard deviation of each
    parameter. 

    The extreme values of the bounds are the transformed parameter values
    1 standard deviation from the mean in unconstrained space. These bounds
    are not representative of parameter uncertainty when using the Ensemble
    Kalman Inversion, but they are representative of the parameter sensitivities
    in the case of Unscented Kalman Inversion.

    Args:
     - data_paths: List of diagnostic data paths from which to draw
      the parameter evolutions.
    """
    tab_colors = get_tab_colors()
    # Fetch parameter and covariance evolutions
    param_evolutions = []
    param_low_evolutions = []
    param_upp_evolutions = []

    for data_path in data_paths:
        phi_mean = dt.ncFetch(data_path, 'ensemble_diags', 'phi_mean')
        phi_low = dt.ncFetch(data_path, 'ensemble_diags', 'phi_low_unc')
        phi_upp = dt.ncFetch(data_path, 'ensemble_diags', 'phi_upp_unc')
        param_evolutions.append(phi_mean)
        param_low_evolutions.append(phi_low)
        param_upp_evolutions.append(phi_upp)
    params = dt.ncFetch(data_path, 'particle_diags', 'param')
    for (i, param) in enumerate(params):
        title = "Parameter "+str(param)
        fig = plt.figure(title)
        max_t = 0
        for c, (param_evol, low_evol, upp_evol, data_path) in enumerate(zip(param_evolutions,
                param_low_evolutions, param_upp_evolutions, data_paths)):
            shading = convert2rgb(tab_colors[c], 0.4)
            t = dt.ncFetch(data_path, 'ensemble_diags', 'iteration')
            max_t = max(t[-1], max_t)
            mean_val = param_evol[:, i]
            lower_bound = low_evol[:, i]
            upper_bound = upp_evol[:, i]
            plt.plot(t, mean_val, color=tab_colors[c])
            plt.fill_between(t, mean_val, lower_bound, color=shading)
            plt.fill_between(t, mean_val, upper_bound, color=shading)
            label = (data_path).split('_')[-1]
        plt.xlabel(r'Iteration')
        plt.ylabel(param)
        plt.xlim(0, max_t)
        ax = fig.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(frameon=False)
        plt.savefig(title+'.pdf', format='pdf')
    return
