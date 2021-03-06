import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import xarray as xr
import os
g = 9.81 # gravitational acceleration

def average_end_ds(ds:xr.Dataset, t_interval_from_end_s = 3 * 3600):
    """ Average dataset from `end - t_interval_from_end_s` to `end` of simulation.
    Args:
     - ds :: xarray.Dataset
     - t_interval_from_end_s :: int - seconds from end of simulation to average over
    """
    ds = ds.where(ds.t > (ds.t[-1] - t_interval_from_end_s).item(), drop = True)
    return ds.mean(dim = 't')

def preprocess_stats(stats_path:str, t_interval_from_end_s = 3 * 3600, drop_zero_area = True):
    """ Preprocess stats to average over `t_interval_from_end_s` seconds from end of simulation.
    Args:
     - stats_path :: str - path to stats.nc
     - t_interval_from_end_s :: int - seconds from end of simulation to average over
     - drop_zero_area :: bool - whether to vertical grid points with zero `updraft_area`
    """
    profiles_ds = xr.open_dataset(stats_path, group = "profiles")
    profiles_ds = profiles_ds.interp(zf = profiles_ds.zc.values).drop("zf").rename({"zf":"zc"}) # interp to cell center
    if drop_zero_area:
        profiles_ds = profiles_ds.where(profiles_ds["updraft_area"] > 0., drop = True)

    timeseries_ds = xr.open_dataset(stats_path, group = "timeseries")
    timeseries_ds = timeseries_ds.isel(t = slice(1,(len(timeseries_ds.t) - 1)))
    reference_ds = xr.open_dataset(stats_path, group = "reference")
    if t_interval_from_end_s:
        profiles_ds = average_end_ds(profiles_ds, t_interval_from_end_s)
        timeseries_ds = average_end_ds(timeseries_ds, t_interval_from_end_s)
    return (profiles_ds, timeseries_ds, reference_ds)


def compute_pi_groups(profiles_ds:xr.Dataset, timeseries_ds:xr.Dataset, reference_ds:xr.Dataset, namelist:dict):
    ''' Compute pi groups given datasets with relevant variables.

    Inputs
     - profiles_ds :: xr.Dataset - Dataset containing profiles group of `stats.nc` file.
     - timeseries_ds :: xr.Dataset - Dataset containing timeseries group of `stats.nc` file.
     - reference_ds :: xr.Dataset - Dataset containing reference group of `stats.nc` file.
     - namelist :: Dict - Dictionary corresponding to namelist.

    Returns
     - ds_out :: xr.Dataset - Dataset containing pi groups.
    '''

    pi_groups = namelist["turbulence"]["EDMF_PrognosticTKE"]["entr_pi_subset"]
    norm_consts = namelist["turbulence"]["EDMF_PrognosticTKE"]["pi_norm_consts"]

    ds_out = xr.Dataset()
    delta_w = profiles_ds['updraft_w'] - profiles_ds['env_w']
    delta_RH = profiles_ds['updraft_RH'] - profiles_ds['env_RH']
    delta_B = profiles_ds['updraft_buoyancy'] - profiles_ds['env_buoyancy']

    if 1 in pi_groups:
        ds_out["pi_1"] = (profiles_ds.zc * delta_B) / (delta_w**2 + timeseries_ds['wstar']**2) / norm_consts[0]
    if 2 in pi_groups:
        ds_out["pi_2"] = (profiles_ds['tke_mean'] - profiles_ds['env_area'] * profiles_ds['env_tke']) / profiles_ds['tke_mean'] / norm_consts[1]
    if 3 in pi_groups:
        ds_out["pi_3"] = np.sqrt(profiles_ds['updraft_area']) / norm_consts[2]
    if 4 in pi_groups:
        ds_out["pi_4"] = delta_RH / norm_consts[3]
    if 5 in pi_groups:
        ds_out["pi_5"] =  profiles_ds.zc / timeseries_ds["Hd"] / norm_consts[4]
    if 6 in pi_groups:
        ref_scale_height = reference_ds["p0_c"] / (reference_ds["??0_c"] * g)
        ds_out["pi_6"] =  profiles_ds.zc / ref_scale_height / norm_consts[5]

    return ds_out


def plot_pi_entr(ds, aux_field_vals:List[float] = None, shade_cloud_frac:bool = True, save_file_name = "fig.png", ylims = None, save_figs = False):
    ''' Create 3-panel plot with pi groups, fractional entrainment/detrainment, and non-dimensional entrainment/detrainment given datasets with relevant variables.

    Inputs
     - profiles_ds :: xr.Dataset - Dataset containing pi groups as data fields (or optionally cloud_fraction).
     - aux_field_vals :: List - List of values to plot horizontal lines at on the y-axis (ie. at cloud base, updraft top)
     - shade_cloud_frac :: bool - If True, shade cloud fraction.

    '''
    if not ylims:
        ylims = [0, ds["zc"].max() + 0.1*ds["zc"].max()]
    linewidth = 3
    plt.figure(figsize = (16,5))
    ax1 = plt.subplot(131)
    if "pi_1" in ds:
        plt.plot(ds["pi_1"].values, ds.zc.values, c = 'k', label = '$\Pi_{1}$', linewidth = linewidth)
    if "pi_2" in ds:
        plt.plot(ds["pi_2"].values, ds.zc.values, label = '$\Pi_{2}$', linewidth = linewidth)
    if "pi_3" in ds:
        plt.plot(ds["pi_3"].values, ds.zc.values, label = '$\Pi_{3}$', linewidth = linewidth)
    if "pi_4" in ds:
        plt.plot(ds["pi_4"].values, ds.zc.values, label = '$\Pi_{4}$', linewidth = linewidth)
    if "pi_5" in ds:
        plt.plot(ds["pi_5"].values, ds.zc.values, label = '$\Pi_{5}$', linewidth = linewidth)
    if "pi_6" in ds:
        plt.plot(ds["pi_6"].values, ds.zc.values, label = '$\Pi_{6}$', linewidth = linewidth)
    if aux_field_vals:
        for val in aux_field_vals:
            plt.axhline(y = val, color = 'k', linestyle = '--', alpha = 0.5)
    if shade_cloud_frac:
        dzh = np.diff(ds["zc"]).mean()/ 2
        cf_max = ds["cloud_fraction"].max().item()
        for cf_i in ds["cloud_fraction"]:
            if ~cf_i.isnull():
                alpha = (cf_i.item() / cf_max) * 0.75
                plt.fill_between([0, 1], cf_i["zc"].item() - dzh, cf_i["zc"].item() + dzh, alpha=alpha, color='gray')

    plt.xlim([-0.05, 1])

    plt.legend()

    plt.xticks(rotation = 45)
    plt.ylabel('Height [m]')
    plt.title('$\Pi$ Groups', weight = 'bold')
    plt.ylim(ylims)
    plt.tight_layout()


    ax2 = plt.subplot(132)
    plt.plot(ds["entrainment_sc"].values, ds.zc.values, label = '$\epsilon_{dyn}$', color = 'b', linewidth = linewidth)
    plt.plot(ds["turbulent_entrainment"].values, ds.zc.values, label = '$\epsilon_{turb}$', color = 'b', linestyle = '--', linewidth = linewidth)
    plt.plot(ds["detrainment_sc"].values, ds.zc.values, label = '$\delta_{dyn}$', color = 'r', linewidth = linewidth)
    if aux_field_vals:
        for val in aux_field_vals:
            plt.axhline(y = val, color = 'k', linestyle = '--', alpha = 0.5)

    plt.ylim(ylims)
    ax2.axes.get_yaxis().set_visible(False)
    plt.legend()
    plt.title("Entrainment/Detrainment Rate [$m^{-1}$]", weight = 'bold')
    plt.xlabel('Entrainment/Detrainment Rate [$m^{-1}$]', weight = 'bold')


    ax3 = plt.subplot(133)
    plt.plot(ds["nondim_entrainment_sc"].values, ds.zc.values, label = '$\epsilon_{nondim}$', color = 'b', linewidth = linewidth)
    plt.plot(ds["nondim_detrainment_sc"].values, ds.zc.values, label = '$\delta_{nondim}$', color = 'r', linewidth = linewidth)
    if aux_field_vals:
        for val in aux_field_vals:
            plt.axhline(y = val, color = 'k', linestyle = '--', alpha = 0.5)

    plt.ylim(ylims)
    ax3.axes.get_yaxis().set_visible(False)
    plt.legend()
    plt.title("Nondimensional Entrainment/Detrainment", weight = 'bold')
    plt.xlabel("Nondimensional Entrainment/Detrainment", weight = 'bold')
    plt.ylabel('Height [m]')

    if save_file_name:
        dir_name = os.path.dirname(save_file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_file_name, dpi = 200)