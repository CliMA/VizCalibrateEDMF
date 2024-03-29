import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import xarray as xr
import os
g = 9.81 # gravitational acceleration

# TC field name : LES field name map
FIELD_MAP = {'cloud_fraction':'cloud_fraction',
    'ql_mean': 'ql_mean',
    'qt_mean': 'qt_mean',
    's_mean' : 's_mean',
    "total_flux_qt": 'qt_flux_z',
    'total_flux_s': 's_flux_z',
    'thetal_mean': 'thetali_mean',
    'updraft_area': 'updraft_area',
    'u_mean': 'u_translational_mean',
    'v_mean': 'v_translational_mean',
    'tke_mean': 'tke_nd_mean',
    'updraft_buoyancy': 'buoyancy_mean',
    'updraft_w': 'w_core'}

# LES field name : TC field name map
FIELD_MAP_R = {v: k for k, v in FIELD_MAP.items()}

ZERO_BOUNDED = ("cloud_fraction", "updraft_area", "qt_mean", "ql_mean",)
TIMESERIES_MEAN_DEFAULT_FIELDS = ['s_mean', 'ql_mean', 'qt_mean', 'thetal_mean', 'total_flux_qt', 'total_flux_s', 'tke_mean', 'updraft_buoyancy', 'updraft_w']
TIMESERIES_UPDRAFT_DEFAULT_FIELDS =['updraft_area', 'updraft_thetal', 'updraft_w', 'updraft_qt', 'updraft_ql', 'updraft_buoyancy', 'entrainment_ml', 'massflux', 'detrainment_ml']
LES_TIMESERIES_MEAN_DEFAULT_FIELDS = ['cloud_fraction', 'ql_mean', 'qt_mean', 's_mean', 'total_flux_qt', 'total_flux_s', 'thetal_mean', 'u_mean', 'updraft_w']


def average_end_ds(ds:xr.Dataset, t_interval_from_end_s = 3 * 3600):
    """ Average dataset from `end - t_interval_from_end_s` to `end` of simulation.
    Args:
    - ds :: xarray.Dataset
    - t_interval_from_end_s :: int - seconds from end of simulation to average over
    """
    ds = ds.where(ds.t > (ds.t[-1] - t_interval_from_end_s).item(), drop = True)
    return ds.mean(dim = 't')

def preprocess_stats(stats_path:str,
                    t_interval_from_end_s = 3 * 3600,
                    interp_z = None,
                    drop_zero_area = False,
                    rectify_surface_fluxes = False):
    """ Preprocess stats to average over `t_interval_from_end_s` seconds from end of simulation.
    Args:
    - stats_path :: str - path to stats.nc
    - t_interval_from_end_s :: int - seconds from end of simulation to average over
    - drop_zero_area :: bool - whether to vertical grid points with zero `updraft_area`
    """
    profiles_ds = xr.open_dataset(stats_path, group = "profiles")

    if "z" in profiles_ds:
        profiles_ds = profiles_ds.rename({"z":"zc"})

    if not (interp_z is None): # interp to new vertical grid
        profiles_ds = profiles_ds.interp(zc = interp_z)
    elif ("zf" in profiles_ds):
        profiles_ds = profiles_ds.interp(zf = profiles_ds.zc.values).drop("zf").rename({"zf":"zc"}) # interp to cell center

    if drop_zero_area:
        profiles_ds = profiles_ds.where(profiles_ds["updraft_area"] > 0., drop = True)

    timeseries_ds = xr.open_dataset(stats_path, group = "timeseries")
    reference_ds = xr.open_dataset(stats_path, group = "reference")


    if rectify_surface_fluxes:
        profiles_ds = rectify_surface_flux(profiles_ds, timeseries_ds)

    if t_interval_from_end_s:
        profiles_ds = average_end_ds(profiles_ds, t_interval_from_end_s)
        timeseries_ds = average_end_ds(timeseries_ds, t_interval_from_end_s)

    return (profiles_ds, timeseries_ds, reference_ds)

def latent_heat_vapor(T):
    # Adapted from ClimaParameters.jl and Thermodynamics.jl
    cp_l = 4181 # isobaric_specific_heat_liquid
    cp_v = 1859 # isobaric_specific_heat_vapor
    lh_v0 = 2500800 # latent_heat_vaporization_at_reference
    T_0 = 273.16 # thermodynamics_temperature_reference
    Δcp = cp_v - cp_l
    return lh_v0 + Δcp * (T - T_0)

'''
    Sets bottom cell in interpolated flux profile equal to surface flux.
    This is needed for LES profiles since neither the resolved nor the SGS fluxes
    include contributions from the surface flux (otherwise flux goes to zero at the surface).
    Returns new ds profiles Dataset with corrected surface flux after interpolation.
'''
def rectify_surface_flux(profiles_ds, timeseries_ds):

    min_z_index = profiles_ds["zc"].argmin()

    # rectify surface qt flux
    lhf_surface = timeseries_ds["lhf_surface_mean"]
    t_surface = timeseries_ds["surface_temperature"]
    surf_qt_flux = lhf_surface / latent_heat_vapor(t_surface)
    profiles_ds["qt_flux_z"].loc[{'zc': profiles_ds.zc[min_z_index]}] = surf_qt_flux

    # rectify surface s flux
    profiles_ds["s_flux_z"].loc[{'zc': profiles_ds.zc[min_z_index]}] = timeseries_ds["s_flux_surface_mean"]

    return profiles_ds


''' Compute mse between variables in 2 datasets, and return as '''
def compute_mse(ds1, ds2, var_list):
    sq_diff = (ds1[var_list] - ds2[var_list])**2
    return sq_diff.sum(dim = "zc") / len(ds1.zc)

def compute_std_time(profiles_ds:xr.Dataset):
    return profiles_ds.std(dim = 't')

def plot_profiles(ds:xr.Dataset, 
                ds_les:xr.Dataset, profiles_ds_les_std = None, ylims = [0, 4000],
                plot_field_names = ['cloud_fraction', 'ql_mean', 'qt_mean', 
                    's_mean', "total_flux_qt", 'total_flux_s'],
                edmf_label:str = 'EDMF', title:str = None, 
                save_fig_path:str = False):

    nrows, ncolumns = 2,3
    fig , ax = plt.subplots(nrows, ncolumns, sharey = True, figsize = (10,7))
    for field_num in range(len(plot_field_names)):
        tc_field_name = plot_field_names[field_num]
        les_field_name = FIELD_MAP[tc_field_name]

        ax_i = ax[np.unravel_index(field_num, (nrows, ncolumns))]
        # plot SCM
        field_var = ds[tc_field_name]
        if les_field_name in ds_les:
            les_field_var = ds_les[les_field_name]
            left_bound_les_shade = les_field_var - profiles_ds_les_std[les_field_name]
            right_bound_les_shade = les_field_var + profiles_ds_les_std[les_field_name]

            # plot LES variance
            if tc_field_name in ZERO_BOUNDED:
                left_bound_les_shade = left_bound_les_shade.where(left_bound_les_shade >= 0., 0.)

            ax_i.fill_betweenx(ds_les.zc,
                    left_bound_les_shade,
                    right_bound_les_shade,
                    alpha = 0.3)
            # plot LES field
            ax_i.plot(les_field_var, ds_les.zc, c = 'k', linewidth = 1.0, label = 'LES')
            ax_i.scatter(les_field_var, ds_les.zc, s = 2, c = 'k')

        if len(field_var) == len(ds.zc):
            z = ds.zc
        elif len(field_var) == len(ds.zf):
            z = ds.zf

        ax_i.plot(field_var, z,  c = 'r', linewidth = 1.0, label = edmf_label)
        ax_i.scatter(field_var, z,  c = 'r', s = 2)

        ax_i.set_xlabel(tc_field_name, weight = 'bold')
        ax_i.grid()
        if ylims:
            ax_i.set_ylim(ylims)
        for tick in ax_i.get_xticklabels():
            tick.set_rotation(25)
    if title:
        fig.suptitle(title, fontsize="x-large", weight = 'bold')
    plt.tight_layout()

    handles, labels = ax_i.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    if save_fig_path:
        dir_name = os.path.dirname(save_fig_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_fig_path, dpi = 200)
    return (plt.gcf(), plt.gca())

def plot_timeseries(ds:xr.Dataset, ylims = [0, 4000],
                    plot_field_names = TIMESERIES_MEAN_DEFAULT_FIELDS, title:str = None,
                    save_fig_path:str = False):

    cmap = plt.get_cmap("turbo")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 8), sharey=True, sharex=True)

    for field_num in range(len(plot_field_names)):
        plot_field_name = plot_field_names[field_num]
        try:
            ax = axs.flat[field_num]

            #########
            plot_field = ds[plot_field_name]
            plot_field = plot_field.where(plot_field > 0)
            plot_field.T.plot.pcolormesh(ax=ax, cmap=cmap)
            ax.set_ylim(ylims)

            if field_num == 0:
                ax.set_ylabel('Height (m)')

            if title:
                plt.suptitle(title, fontsize=16)
        except:
            print("Could not plot", plot_field_name)

    if save_fig_path:
        dir_name = os.path.dirname(save_fig_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_fig_path, dpi=200)

    plt.tight_layout()
    plt.show()


def get_pi_var_names(namelist:dict):
    pi_groups = namelist["turbulence"]["EDMF_PrognosticTKE"]["entr_pi_subset"]
    return ["pi_{i}".format(i = pi_i) for pi_i in pi_groups]

def get_pi_groups(profiles_ds:xr.Dataset, namelist:dict):
    return profiles_ds[get_pi_var_names(namelist)]


def compute_pi_groups(profiles_ds:xr.Dataset, timeseries_ds:xr.Dataset, reference_ds:xr.Dataset, namelist:dict):
    ''' Compute pi groups given datasets with relevant variables.

    Args:
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
    eps = sys.float_info.epsilon

    if 1 in pi_groups:
        ds_out["pi_1"] = (profiles_ds.zc * delta_B) / (delta_w**2 + eps) / norm_consts[0]
    if 2 in pi_groups:
        ds_out["pi_2"] =  profiles_ds['env_tke'] / (delta_w**2 + eps) / norm_consts[1]
    if 3 in pi_groups:
        ds_out["pi_3"] = np.sqrt(profiles_ds['updraft_area']) / norm_consts[2]
    if 4 in pi_groups:
        ds_out["pi_4"] = delta_RH / norm_consts[3]
    if 5 in pi_groups:
        ds_out["pi_5"] =  profiles_ds.zc / timeseries_ds["Hd"] / norm_consts[4]
    if 6 in pi_groups:
        try:
            ref_scale_height = reference_ds["p0_c"] / (reference_ds["ρ0_c"] * g)
        except:
            ref_scale_height = reference_ds["p_c"] / (reference_ds["ρ_c"] * g)
        ds_out["pi_6"] =  profiles_ds.zc / ref_scale_height / norm_consts[5]

    return ds_out


def plot_pi_entr(ds, aux_field_vals:List[float] = None, shade_cloud_frac:bool = True, 
                save_fig_path = "fig.png", 
                ylims = None, xlims = None, save_figs = False, xlog = False):
    ''' Create 3-panel plot with pi groups, fractional entrainment/detrainment, 
    and non-dimensional entrainment/detrainment given datasets with relevant variables.

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
    for pi_i in range(1,7):
        pi_name = "pi_{pi_i}".format(pi_i = pi_i)
        if pi_name in ds:
            plt.plot(ds[pi_name].values, ds.zc.values, label = '$\Pi_{pi_i}$'.format(pi_i = pi_i), linewidth = linewidth)
    if aux_field_vals:
        for val in aux_field_vals:
            plt.axhline(y = val, color = 'k', linestyle = '--', alpha = 0.5)
    if shade_cloud_frac:
        dzh = np.diff(ds["zc"]).mean()/ 2
        cf_max = ds["cloud_fraction"].max().item()
        for cf_i in ds["cloud_fraction"]:
            if (~cf_i.isnull()) & (cf_max > 0.0):
                alpha = (cf_i.item() / cf_max) * 0.75
                plt.fill_between([-0.1, 1.0], cf_i["zc"].item() - dzh, cf_i["zc"].item() + dzh, alpha=alpha, color='gray')

    plt.xlim([-0.1, 1.0])

    plt.legend()

    plt.xticks(rotation = 45)
    plt.ylabel('Height [m]')
    plt.title('$\Pi$ Groups', weight = 'bold')
    plt.ylim(ylims)
    plt.tight_layout()


    ax2 = plt.subplot(132)
    if "entrainment_sc" in ds:
        plt.plot(ds["entrainment_sc"].values, ds.zc.values, label = '$\epsilon_{dyn}$', color = 'b', linewidth = linewidth)
    if "entrainment_ml" in ds:
        plt.plot(ds["entrainment_ml"].values, ds.zc.values, label = '$\epsilon^{ML}_{dyn}$', color = 'b', linestyle = '--', linewidth = linewidth)
    if "detrainment_sc" in ds:
        plt.plot(ds["detrainment_sc"].values, ds.zc.values, label = '$\delta_{dyn}$', color = 'r', linewidth = linewidth)
    if "detrainment_ml" in ds:
        plt.plot(ds["detrainment_ml"].values, ds.zc.values, label = '$\delta^{ML}_{dyn}$', color = 'r', linestyle = '--', linewidth = linewidth)

    plt.plot(ds["turbulent_entrainment"].values, ds.zc.values, label = '$\epsilon_{turb}$', color = 'gray', linestyle = '--', linewidth = linewidth)
    if aux_field_vals:
        for val in aux_field_vals:
            plt.axhline(y = val, color = 'k', linestyle = '--', alpha = 0.5)

    plt.ylim(ylims)
    if xlims:
        plt.xlim(xlims)
    if xlog:
        plt.xscale("log")
    ax2.axes.get_yaxis().set_visible(False)
    plt.legend()
    plt.title("Entrainment/Detrainment Rate [$m^{-1}$]", weight = 'bold')
    plt.xlabel('Entrainment/Detrainment Rate [$m^{-1}$]', weight = 'bold')


    ax3 = plt.subplot(133)
    if "nondim_entrainment_sc" in ds:
        plt.plot(ds["nondim_entrainment_sc"].values, ds.zc.values, label = '$\epsilon_{nondim}$', color = 'b', linewidth = linewidth)
    if "nondim_entrainment_ml" in ds:
        plt.plot(ds["nondim_entrainment_ml"].values, ds.zc.values, label = '$\epsilon^{ML}_{nondim}$', color = 'b', linestyle = '--', linewidth = linewidth)
    if "nondim_detrainment_sc" in ds:
        plt.plot(ds["nondim_detrainment_sc"].values, ds.zc.values, label = '$\delta_{nondim}$', color = 'r', linewidth = linewidth)
    if "nondim_detrainment_ml" in ds:
        plt.plot(ds["nondim_detrainment_ml"].values, ds.zc.values, label = '$\delta^{ML}_{nondim}$', color = 'r', linestyle = '--', linewidth = linewidth)

    if aux_field_vals:
        for val in aux_field_vals:
            plt.axhline(y = val, color = 'k', linestyle = '--', alpha = 0.5)

    plt.ylim(ylims)
    if xlims:
        plt.xlim(xlims)
    if xlog:
        plt.xscale("log")
    ax3.axes.get_yaxis().set_visible(False)
    plt.legend()
    plt.title("Nondimensional Entrainment/Detrainment", weight = 'bold')
    plt.xlabel("Nondimensional Entrainment/Detrainment", weight = 'bold')
    plt.ylabel('Height [m]')

    if save_fig_path:
        dir_name = os.path.dirname(save_fig_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_fig_path, dpi = 200)
    
    return (plt.gcf(), plt.gca())