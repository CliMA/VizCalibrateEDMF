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

ZERO_BOUNDED = ("cloud_fraction", "updraft_area", "qt_mean", "ql_mean",)

PLOT_NAME_MAP = {"s_mean": r"Entropy $[J \cdot K^{-1}]$",
                "ql_mean": r"Liq Water Specific Humidity $[kg \cdot kg^{-1}]$", 
                "qt_mean": r"Total Water Specific Humidity $[kg \cdot kg^{-1}]$", 
                "total_flux_qt" : r"Total Water Specific Humidty Flux $[kg \cdot m \cdot kg^{-1} \cdot s^{-1}$]", 
                "total_flux_s": r"Total Entropy Flux $[J \cdot m \cdot s^{-1} \cdot K^{-1}]$",
                "lwp_mean": r"Liquid Water Path $[(kg \cdot m^{-2})]$"}

PLOT_NAME_MAP_ABBREVIATED = {
    "s_mean": r"$\mathbf{\bar{s}}$ $[J \cdot K^{-1}]$",
    "ql_mean": r"$\mathbf{\bar{q_l}}$ $[kg \cdot kg^{-1}]$",
    "qt_mean": r"$\mathbf{\bar{q_t}}$ $[kg \cdot kg^{-1}]$",
    "total_flux_qt": r"$\mathbf{\overline{w'q_t'}}$ $[m \cdot s^{-1}]$",
    "total_flux_s": r"$\mathbf{\overline{w's'}}$ $[J \cdot m \cdot s^{-1} \cdot K^{-1}]$",
    "lwp_mean": r"$\mathbf{LWP}$ $[kg \cdot m^{-2}]$"
}

PLOT_NAME_MAP_SYMBOLS = {
    "s_mean": r"$\mathbf{\bar{s}}$",
    "ql_mean": r"$\mathbf{\bar{q_l}}$",
    "qt_mean": r"$\mathbf{\bar{q_t}}$",
    "total_flux_qt": r"$\mathbf{\overline{w'q_t'}}$",
    "total_flux_s": r"$\mathbf{\overline{w's'}}$",
    "lwp_mean": r"$\mathbf{LWP}$"
}



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

def compute_offline_mse_from_unpacked_profiles(ds_g, ds_y, ds_diagnostics,  num_iterations = 50, num_variables = 6, val_bool = False):
    
    num_particles = len(ds_diagnostics.particle)

    mse_data = xr.DataArray(np.nan, dims=("iteration", "variable"), coords={"iteration": range(1, num_iterations), "variable": range(num_variables)})
    mse_by_particle_data = xr.DataArray(coords={"iteration": range(1, num_iterations), "variable": range(num_variables), "particle": range(num_particles)}, dims=["iteration", "variable", "particle"])

    # number of cases in iteration
    if val_bool: 
        iter_num_cases = len(ds_diagnostics.batch_index_val)
    else:
        iter_num_cases = len(ds_diagnostics.batch_index)

    for iteration in range(1, num_iterations):
        if val_bool:
            y_sub = ds_y
        else:
            batch_inds_i = ds_diagnostics["batch_indices"].sel(iteration = iteration)
            y_sub = ds_y.sel(case = batch_inds_i.values)
        ds_sub = ds_g.sel(iteration = iteration).isel(case = slice(0,iter_num_cases))

        # vertical profiles 
        for variable in range(5):
            n_by_var = 0
            diff_squared_sum = 0 

            n_by_var_by_particle = np.zeros(num_particles)
            diff_squared_sum_by_particle =  np.zeros(num_particles)
            # compute mse averaged over particles 
            for case_i in range(iter_num_cases):
                y_true = y_sub["all_y"].isel(variable = variable, case = case_i)
                y_pred = ds_sub["all_g"].isel(variable = variable, case = case_i)

                # difference by ensemble member
                diff_squared_ds = (y_pred - y_true)**2

                # compute average mse across ensemble members for iteration, variable
                diff_squared = diff_squared_ds.values.flatten()
                diff_squared_sum += np.nansum(diff_squared)
                # count nonnans
                non_nan_elements = np.sum(~np.isnan(diff_squared))
                n_by_var += non_nan_elements


                diff_squared_sum_by_particle += diff_squared_ds.sum("dof", skipna = True).values
                # count nonnans
                n_by_var_by_particle += diff_squared_ds.count(dim = 'dof').values


            mse_var_iter = np.sqrt(diff_squared_sum / n_by_var)
            mse_data.loc[iteration, variable] = mse_var_iter


            n_by_var_by_particle[n_by_var_by_particle == 0] = np.nan

            mse_var_iter_by_particle =  np.sqrt(diff_squared_sum_by_particle / n_by_var_by_particle)
            mse_by_particle_data.loc[dict(iteration=iteration, variable=variable)] = mse_var_iter_by_particle


        # process integrated quantities 
        y_true_int = ds_sub["all_g_integrated"].isel(variable = -1)
        y_pred_int = y_sub["all_y_integrated"].isel(variable = -1)
        y_pred_int = y_pred_int.assign_coords(case = y_true_int["case"])
        diff_squared_ds = (y_pred_int - y_true_int)**2
        diff_squared_ds = diff_squared_ds.isel(integrated = 0)

        mse_data.loc[iteration, 5] = diff_squared_ds.mean(dim = ["particle", "case"], skipna = True)

        mse_by_particle = diff_squared_ds.mean(dim='case', skipna = True)
        mse_by_particle_data.loc[dict(iteration=iteration, variable=5)] = mse_by_particle
    
    return (mse_data, mse_by_particle_data)




def plot_var_offline(mse_list,
                mse_by_particle_list,
                nrows:int = 6,
                ncols:int = 1,
                val_rmse:bool = False,
                var_names = None,
                box_and_whiskers:bool = True,
                max_min_shading:bool = True, 
                plot_labels:list = None,
                curve_colors = ['red', 'blue'],
                ylims = {"s_mean": [0, 30], "qt_mean": [0, 1e-7], "ql_mean": [0, 2e-4], "total_flux_s": [0, 0.3], "total_flux_qt": [0, 7e-5], "lwp_mean": [0, 0.2]},
                baseline_hz_line:bool = None,
                save_fig_path:str = False, 
                average_epochs: bool = False,
                display_epoch:bool = True,
                iterations_per_epoch: int = None,
                plot_name_map:dict = None, 
                ylabel_bool = True,
                xlims = None,
                suptitle = None, 
                linewidth=2,
                legend_labels = ["NN", "Linreg", "Cohen et al., 2020",],
                ):
    
    fig, axs = plt.subplots(nrows, ncols, sharex = True, figsize=(ncols * 6, nrows * 3))
    if nrows > 1 and ncols > 1:
        axs = axs.flatten()

    for var_name_i, ax in zip(range(len(var_names)), axs):
        var_name = var_names[var_name_i]
        baseline_added = False
        lines = []
        for mse_list_i in range(len(mse_list)):
            u_dataframe = mse_list[mse_list_i].isel(variable  = var_name_i).to_dataframe(name = "mse").reset_index()

            u_dataframe["Epoch"] = (u_dataframe["iteration"] / iterations_per_epoch).round(1)

            u_dataframe_by_part = mse_by_particle_list[mse_list_i].isel(variable  = var_name_i).to_dataframe(name = "mse_by_particle").reset_index()
            u_dataframe_by_part["Epoch"] = (u_dataframe_by_part["iteration"] / iterations_per_epoch).round(1)

            if box_and_whiskers:
                if display_epoch:
                    x = u_dataframe_by_part["Epoch"]
                else:
                    x = u_dataframe_by_part['iteration']
        
                box = seaborn.boxplot(x=x,
                                y=u_dataframe_by_part["mse_by_particle"],
                                data=u_dataframe_by_part[['Epoch', "mse_by_particle"]],
                                ax=ax,
                                width=0.5,
                                color=curve_colors[mse_list_i],
                                linewidth = 1,
                                boxprops=dict(alpha=0.25),
                                showfliers=False,)

            
            if max_min_shading:
                mean_max = u_dataframe_by_part.groupby('iteration')['mse_by_particle'].agg(['min', 'max']).reset_index()
                mean_max["Epoch"] = (mean_max["iteration"]/ iterations_per_epoch).round()

                x = mean_max['iteration'] - 1

                ax.fill_between(x.values, mean_max["min"].values, mean_max["max"].values, alpha = 0.1,  color=curve_colors[mse_list_i])
                

            # get mean mse across ensemble members
            mean_values = u_dataframe["mse"]

            if not (plot_labels is None):
                plot_label_i = plot_labels[mse_list_i]
            else:
                plot_label_i = None

            line, = ax.plot(mean_values.values, linewidth=linewidth, label=plot_label_i, color=curve_colors[mse_list_i])
            lines.append(line)
            if not baseline_added and not (baseline_hz_line is None) & (mse_list_i == len(mse_list) - 1):
                line_baseline = ax.axhline(y=baseline_hz_line[var_name], color='grey', linestyle='--', label = "Cohen et al., 2020")
                baseline_added = True

            if not (plot_name_map is None) and ylabel_bool:
                ax.set_ylabel(plot_name_map[var_name], weight = "bold", fontsize = 17)
            else:
                ax.set_ylabel(" ", weight="bold", fontsize=17)

            ax.set_ylim(ylims[var_name])

            if xlims:
                ax.set_xlim(xlims)

        if (var_name_i == 0) and (not legend_labels is None):
            lines = [lines[0], lines[1], line_baseline]
            ax.legend(lines, legend_labels, prop={'weight': 'bold'})
        else:
            ax.legend().set_visible(False)

        
        
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.tick_params(axis='y', labelsize=12)

        if var_name_i == len(var_names) - 1:
            if display_epoch:
                ax.set_xlabel('Epoch', weight='bold', fontsize = 17)
            else:
                ax.set_xlabel('Iteration', weight='bold', fontsize = 17)

    if display_epoch and iterations_per_epoch:
        max_iteration = u_dataframe["iteration"].max()
        epochs = np.arange(0, max_iteration + iterations_per_epoch, iterations_per_epoch)
        epoch_labels = epochs // iterations_per_epoch
        # Get the last axis for labeling and ticks
        last_ax = np.atleast_1d(axs)[-1]
        last_ax.set_xticks(epochs[:-1])
        last_ax.set_xticklabels(epoch_labels[:-1])
        last_ax.set_xlim(epochs[0], epochs[-2])
        last_ax.set_xlabel('Epoch')

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.draw()
    if save_fig_path:
        dir_name = os.path.dirname(save_fig_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_fig_path, dpi=200)

def plot_var_mse(ds_dict,
                nrows:int = 3,
                ncols:int = 2,
                val_rmse:bool = False,
                var_names = None,
                stripplot_bool:bool = False,
                box_and_whiskers:bool = False,
                plot_labels:list = None,
                curve_colors = ['red', 'blue'],
                ylims = {"s_mean": [0, 20], "qt_mean": [0, 1e-2], "ql_mean": [0, 2e-4], "total_flux_s": [0, 0.3], "total_flux_qt": [0, 7e-5], "lwp_mean": [0, 0.2]},
                baseline_hz_line:bool = None,
                save_fig_path:str = False, 
                average_epochs: bool = False,
                display_epoch:bool = True,
                iterations_per_epoch: int = None,
                plot_name_map:dict = None, 
                ylabel_bool = True,
                xlims = None,
                suptitle = None, 
                ):
    
    legend_labels = []

    if isinstance(ds_dict, dict):
        ds_dict = [ds_dict, ]
    if var_names is None:
        var_names = ds_dict[0]["reference"]["ref_variable_names"].isel(config = 0).values

    fig, axs = plt.subplots(nrows, ncols, sharex = True, figsize=(ncols * 8, nrows * 4))
    if nrows > 1 and ncols > 1:
        axs = axs.flatten()

    for var_name_i, ax in zip(range(len(var_names)), axs):
        var_name = var_names[var_name_i]
        if val_rmse:
            mse_field_name = "val_mse_{}_full".format(var_name)
        else:
            mse_field_name = "rmse_{}_full".format(var_name)
        for ds_dict_i in range(len(ds_dict)):
            u_dataframe = ds_dict[ds_dict_i]["particle_diags"][mse_field_name].to_dataframe().reset_index()
            u_dataframe["Epoch"] = (u_dataframe["iteration"] / iterations_per_epoch).round(1)
            if stripplot_bool:
                if display_epoch:
                    x = u_dataframe["Epoch"]
                else:
                    x = u_dataframe['iteration']

                seaborn.stripplot(x =  x,
                                    y = u_dataframe[mse_field_name],
                                    data = u_dataframe[['iteration', mse_field_name]],
                                    ax = ax,
                                    alpha = 0.2,
                                    size = 2.0)

            if box_and_whiskers:
                if display_epoch:
                    x = u_dataframe["Epoch"]
                else:
                    x = u_dataframe['iteration']
        
                box = seaborn.boxplot(x=x,
                                y=u_dataframe[mse_field_name],
                                data=u_dataframe[['iteration', mse_field_name]],
                                ax=ax,
                                width=0.5,
                                color=curve_colors[ds_dict_i],
                                linewidth = 1,
                                boxprops=dict(alpha=0.25),
                                showfliers=False,)

            u_dataframe["iteration"] = u_dataframe["iteration"] - 1

            # get mean mse across ensemble members
            mean_values = u_dataframe.groupby('iteration')[mse_field_name].mean()
            if average_epochs and iterations_per_epoch:
                # Apply rolling mean without centering
                u_dataframe.set_index('iteration', inplace=True)
                mean_values = mean_values.rolling(window=iterations_per_epoch, min_periods=1, center=False).mean()

            if not (plot_labels is None):
                plot_label_i = plot_labels[ds_dict_i]
            else:
                plot_label_i = None

            line, = ax.plot(mean_values.values, linewidth=2, label=plot_label_i, color=curve_colors[ds_dict_i])

            if not (baseline_hz_line is None):
                print(baseline_hz_line[var_name])
                ax.axhline(y=baseline_hz_line[var_name], color='r', linestyle='--')

            max_val = u_dataframe[mse_field_name].max()
            ax.set_ylim(0, max_val)

            if not (plot_name_map is None) and ylabel_bool:
                ax.set_ylabel(plot_name_map[var_name], weight = "bold")
            else:
                ax.set_ylabel("", weight = "bold")

            ax.set_ylim(ylims[var_name])

            if xlims:
                ax.set_xlim(xlims)

        if var_name_i == 0:
            ax.legend()
        else:
            ax.legend().set_visible(False) # Explicitly hide legend for other subplots
        
        plt.tight_layout()
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=12))

        if display_epoch:
            ax.set_xlabel('Epoch', weight='bold')
        else:
            ax.set_xlabel('Iteration', weight='bold')

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')

    plt.draw()
    if save_fig_path:
        dir_name = os.path.dirname(save_fig_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_fig_path, dpi=200)


def construct_profiles(ds_dict:dict, 
                       iteration:int = 1, 
                       val_bool:bool = False,
                       batching = False, 
                       integrated_quantities:list = ['lwp_mean']):
    """
    Unpack profiles (and integrated quanities like LWP) from g_full and y_full vectors.
    """

    if val_bool:
        ref_name = "ref_variable_names_val"
        dof_name = "dof_val"
        norm_factor_name = "norm_factor_val"
        config_name = "config_val"
        config_z_obs_name = "config_z_obs_val"
        y_full_name = "y_full_val"
        g_full_name = "val_g_full"
    else:
        ref_name = "ref_variable_names"
        dof_name = "dof"
        norm_factor_name = "norm_factor"
        config_name = "config"
        config_z_obs_name = "config_z_obs"
        y_full_name = "y_full"
        g_full_name = "g_full"

    ds_ref = ds_dict["reference"]
    ds_particle = ds_dict["particle_diags"]
    var_names = ds_ref[ref_name].isel(**{config_name: 0}).values
    num_vars = len(var_names)
    num_cases = len(ds_ref[config_name])

    num_particles = len(ds_particle.particle)
    dof = len(ds_ref[dof_name])
    norm_factor = ds_ref[norm_factor_name]

    all_g = np.zeros((num_cases, num_vars, num_particles, dof)) #(num_cases, num_vars, num_particles, num_grid_points)
    all_g_integrated = np.zeros((num_cases, num_vars, num_particles, 1))
    all_y = np.zeros((num_cases, num_vars, dof))
    all_y_integrated = np.zeros((num_cases, num_vars, 1))

    print("Constructing profiles for iteration ", iteration)

    if not batching: 
        idx_y_full = 0
        for case_num in range(num_cases):
            print("Processing case: ", case_num)
            for var_ind in range(ds_ref.num_vars[case_num].item()):
                var_name = var_names[var_ind]
                if var_name in integrated_quantities:
                    dof = 1
                else:
                    dof = np.count_nonzero(ds_ref[config_z_obs_name].isel(**{config_name: case_num}))

                y_full_train = ds_ref[y_full_name].values[idx_y_full:idx_y_full+int(dof)]*np.sqrt(norm_factor[var_ind, case_num]).item()
                if dof != 1:
                    all_y[case_num, var_ind, :dof] = y_full_train
                elif dof == 1:
                    all_y_integrated[case_num, var_ind, :] = y_full_train
                for particle_num in range(num_particles):
                    g_y_val = ds_particle[g_full_name].isel(particle = particle_num, iteration = iteration).values[idx_y_full:idx_y_full+int(dof)]*np.sqrt(norm_factor[var_ind, case_num]).item()
                    if dof != 1:
                        all_g[case_num, var_ind, particle_num, :dof] = g_y_val
                    elif dof == 1:
                        all_g_integrated[case_num, var_ind, particle_num, :] = g_y_val

                idx_y_full += int(dof)


    else:
        idx_y_full = 0
        batch_size = len(ds_particle["batch_index"])
        batch_inds_iter_i = ds_particle["batch_indices"].sel(iteration = iteration)
        for case_num in range(batch_size):
            print("Processing case: ", case_num)
            for var_ind in range(ds_ref.num_vars[case_num].item()):
                var_name = var_names[var_ind]
                if var_name in integrated_quantities:
                    dof = 1
                else:
                    dof = np.count_nonzero(ds_ref[config_z_obs_name].isel(**{config_name: case_num}))

                y_full_train = ds_ref[y_full_name].values[idx_y_full:idx_y_full+int(dof)]*np.sqrt(norm_factor[var_ind, case_num]).item()
                if dof != 1:
                    all_y[case_num, var_ind, :dof] = y_full_train
                elif dof == 1:
                    all_y_integrated[case_num, var_ind, :] = y_full_train
                for particle_num in range(num_particles):
                    g_y_val = ds_particle[g_full_name].isel(particle = particle_num, iteration = iteration).values[idx_y_full:idx_y_full+int(dof)]*np.sqrt(norm_factor[var_ind, case_num]).item()
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
                    ax = None,
                    val_bool:bool = False, 
                    sigma = False, 
                    row_ind = None, 
                    integrated_quantities:list = ['lwp_mean'],
                    iteration = None, 
                    save_fig_path:str = False):

    """Plot profiles (and integrated quanities like LWP) from dict created from `construct_profiles`"""

    if val_bool:
        ref_name = "ref_variable_names_val"
        z_obs_var_name = "config_z_obs_val"
        config_name = "config_val"
        config_sup = "config_name_val"
        norm_factor_name = "norm_factor_val"
        config_field = "config_field_val"
    else:
        ref_name = "ref_variable_names"
        z_obs_var_name = "config_z_obs"
        config_name = "config"
        config_sup = "config_name"
        norm_factor_name = "norm_factor"
        config_field = "config_field"



    ds_ref = ds_dict["reference"]
    var_names = ds_ref[ref_name].isel(**{config_name: 0}).values
    norm_factors = ds_ref[norm_factor_name]

    num_vars = len(var_names)
    z_val = ds_ref[z_obs_var_name].values[:,case_i]

    if ax is None:
        fig , ax = plt.subplots(nrows, ncols, sharey = True, figsize = (8,12))

    if nrows == 1 or ncols == 1:
        ax = np.array(ax).reshape(nrows, ncols)

    for var_i in range(y_dict["all_g"].shape[1]):

        if not (row_ind is None):
            ax_i = ax[row_ind, var_i]
        else:
            ax_i = ax[np.unravel_index(var_i, (nrows, ncols))]
        var_name = var_names[var_i]
        ds_kwargs = {config_name: case_i}
        z_case_i = ds_ref[z_obs_var_name].isel(**ds_kwargs)
        if var_name in integrated_quantities:
            y_val = y_dict["all_y_integrated"][case_i, var_i,:].item()
            ax_i.axvline(x=y_val, color='k', linestyle='-', label = "LES")
            x_var = y_dict["all_g_integrated"][case_i, var_i,:]
            ax_i.scatter(x_var, 1000*np.ones(len(x_var)), s = 10, alpha = 0.3)

            ds_kwargs = {config_name: case_i, config_field : var_i}
            var_variance = norm_factors.isel(**ds_kwargs).item()
            left_bound_les_shade = y_val - 2*np.sqrt(var_variance)
            right_bound_les_shade = y_val + 2*np.sqrt(var_variance)

            ax_i.fill_betweenx(y_val*np.ones(80),
                    left_bound_les_shade*np.ones(80),
                    right_bound_les_shade*np.ones(80),
                    alpha = 0.7,
                    label = r"noise")

        else:
            dof = np.count_nonzero(z_case_i)
            y_prof = y_dict["all_y"][case_i, var_i, :dof]
            ax_i.plot(y_dict["all_g"][case_i, var_i,:,:dof].T, z_val[:dof], alpha = 0.1)
            ax_i.plot(y_prof, z_val[:dof], c = 'k', label = "LES")

            ds_kwargs = {config_name: case_i, config_field : var_i}
            left_bound_les_shade = y_prof - 2*np.sqrt(norm_factors.isel(**ds_kwargs).item())
            right_bound_les_shade = y_prof + 2*np.sqrt(norm_factors.isel(**ds_kwargs).item())
            print(var_i, norm_factors.isel(**ds_kwargs).item())
            # plot LES variance
            if var_name in ZERO_BOUNDED:
                left_bound_les_shade[left_bound_les_shade < 0] = 0

            ax_i.fill_betweenx(z_val[:dof],
                    left_bound_les_shade,
                    right_bound_les_shade,
                    alpha = 0.7,
                    label = r"noise")

        ax_i.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_i.xaxis.get_offset_text().set_fontsize(12)

        if row_ind == 0:
            ax_i.set_xticks([])
        if row_ind == 1:
            ax_i.tick_params(axis='x', rotation=20)
            ax_i.set_xlabel(PLOT_NAME_MAP_ABBREVIATED[var_names[var_i]], weight = "bold", fontsize = 17)
            ax_i.xaxis.set_label_coords(0.5, -0.20)
        
        if (var_i == 0) & (not iteration is None):
            ax_i.set_ylabel("Iteration {iteration}".format(iteration = iteration), weight = "bold", fontsize = 17)

        if (row_ind == 0) & (var_i == 4):
            ax_i.legend(loc = "upper right", fontsize=10, prop={'weight': 'bold'})

        ax_i.set_ylim(0, 2000)
        padding = 0.1*(y_prof.max().item() -  y_prof.min().item())
        ax_i.set_xlim(y_prof.min().item() - padding , y_prof.max().item() + padding)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)

    if save_fig_path:
        dir_name = os.path.dirname(save_fig_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_fig_path, dpi=200)


def plot_y_profiles_iter(ds_dict_full, 
                        iterations:list = [1, 5], 
                        val_bool:bool = False, 
                        sigma = True,
                        save_fig_dir = "./figs/y_profiles", 
                        case_ind = None, 
                        nrows = None, 
                        ncols = None,):
    if val_bool:
        config_name_var = "config_name_val"
    else:
        config_name_var = "config_name"
    case_names = ds_dict_full["reference"][config_name_var].values

    if (not nrows is None) & (not ncols is None):
        fig , ax = plt.subplots(nrows, ncols, sharey = True, figsize = (14, 8))

    row_ind = 0
    for iteration in iterations:
        profiles = construct_profiles(ds_dict_full, iteration = iteration, val_bool = val_bool)
        if (case_ind is None):
            for case_i in range(len(case_names)):
                case_name_i = case_names[case_i]
                save_fig_path = os.path.join(save_fig_dir, "iter_{}/case{}.pdf".format(iteration, case_i))
                plot_y_profiles(ds_dict_full, 
                                profiles, 
                                case_i = case_i, 
                                val_bool = val_bool, 
                                sigma = sigma, 
                                save_fig_path = save_fig_path, 
                                ax = ax, 
                                nrows = nrows, 
                                ncols = ncols,
                                row_ind = row_ind,
                                iteration = iteration)
        else:
            case_name_i = case_names[case_ind]
            save_fig_path = os.path.join(save_fig_dir, "iter_{}/case{}.pdf".format(iteration, case_ind))
            plot_y_profiles(ds_dict_full, 
                            profiles, 
                            case_i = case_ind, 
                            val_bool = val_bool, 
                            sigma = sigma, 
                            save_fig_path = save_fig_path, 
                            ax = ax, 
                            nrows = nrows, 
                            ncols = ncols,
                            row_ind = row_ind,
                            iteration = iteration)

        row_ind += 1




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



def plot_parameter_evol_violin(ds_dict, save_fig_dir = "./parameter_evol"):
    prior_std = 1.0

    ds_particle = ds_dict["particle_diags"]
    ds_ens = ds_dict["ensemble_diags"]
    ds_prior = ds_dict["prior"]

    for param_i in range(len(ds_particle.param)):

        # plot in constrained (physical) space
        u_dataframe = ds_particle['phi'].isel(param = param_i).to_dataframe().reset_index()
        param_i_name = u_dataframe['param'][1]

        fig, ax = plt.subplots(figsize=(12,5))
        x_box = seaborn.violinplot(x = u_dataframe['iteration'], 
                                y = u_dataframe['phi'], 
                                data = u_dataframe[['iteration', 'phi']], 
                                split=True, hue_order = [False, True], hue = True, alpha = 0.5)


        seaborn.stripplot(x = u_dataframe['iteration'], 
                        y = u_dataframe['phi'], 
                        data = u_dataframe[['iteration', 'phi']], 
                        c = 'k', 
                        alpha = 0.5, 
                        size = 2.0)


        plt.plot(ds_ens['iteration'] - 1, ds_ens['phi_mean'].isel(param = param_i))
        plt.plot(ds_ens['iteration'] - 1, ds_ens['phi_upp_unc'].isel(param = param_i))
        plt.plot(ds_ens['iteration'] - 1, ds_ens['phi_low_unc'].isel(param = param_i))
        plt.ylabel(param_i_name)

        ds_prior_param_i = ds_prior.sel(param = param_i_name)
        # plot prior mean
        plt.axhline(y=ds_prior_param_i["phi_mean_prior"], color='r', linestyle='--', alpha = 0.5)
        plt.axhline(y=ds_prior_param_i["phi_low_unc_prior"], color='r', linestyle='--', alpha = 0.5)
        plt.axhline(y=ds_prior_param_i["phi_upp_unc_prior"], color='r', linestyle='--', alpha = 0.5)

        textstr = " Final: " + str(ds_particle['phi'].isel(param = param_i, iteration = -1).mean().item())
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right', bbox=props)


        if save_fig_dir:
            if not os.path.exists(save_fig_dir):
                os.makedirs(save_fig_dir)
            plt.savefig(os.path.join(save_fig_dir, param_i_name + ".png"), dpi=200)