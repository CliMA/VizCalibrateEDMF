import argparse
from plot_profiles_helper import *
import os
from glob import glob
from src.data_tools import *

"""A command line interface to plot vertically-resolved pi groups and entrainment/detrainment fields, 
    given a path to completed TurbulenceConvection runs. Horizontal dashed lines indicate updraft cloud base and updraft cloud top.
    Shading indicates grid-mean cloud fraction. Pi groups are saved in a `.nc` file. 
    
    Example usage:
        python plot_profiles.py --tc_output_dir=DATA_DIR --save_figs_dir=SAVE_FIGS_DIR
    where DATA_DIR is a directory containing TurbulenceConvection runs, and SAVE_FIGS_DIR is the directory to save the resulting figures.
    """

T_INTERVAL_FROM_END = 12.0 * 3600.0

def main():

    parser = argparse.ArgumentParser(description='Compute or load and save pi groups from TurbulenceConvection runs.')
    parser.add_argument("--tc_output_dir", required=True, help="Directory containing TurbulenceConvection runs.")
    parser.add_argument("--save_figs_dir", help="Directory to save the resulting figures. Defaults to ../figs/<NAME>.")
    parser.add_argument("--diag_path", required=False, help="Directory containing TurbulenceConvection runs.")
    args = parser.parse_args()

    data_dir = args.tc_output_dir
    diag_path = args.diag_path

    # Default save_figs_dir to "../figs/<NAME>" where <NAME> is the last part of the tc_output_dir
    if args.save_figs_dir:
        save_figs_dir = args.save_figs_dir
    else:
        name = os.path.basename(os.path.normpath(data_dir))  # Get the last directory name
        save_figs_dir = os.path.join('..', 'figs', name)

    print(f"Data directory: {data_dir}")
    print(f"Figures will be saved in: {save_figs_dir}")

    profiles = []
    profiles_full = []
    rel_paths = os.listdir(data_dir)
    for rel_path in rel_paths:
        rel_path = os.path.join(data_dir, rel_path)
        stats = stats_path(rel_path)
        namelist = load_namelist(namelist_path(rel_path))

        # plot entrainment/detrainment
        profiles_ds, timeseries_ds, reference_ds = preprocess_stats(stats, t_interval_from_end_s = T_INTERVAL_FROM_END, drop_zero_area = True)
        pi_var_names = get_pi_var_names(namelist)
        # if pi groups saves in TC stats file, use those. Otherwise compute offline
        if any(pi_var_name in profiles_ds for pi_var_name in pi_var_names):
            pi_groups = get_pi_groups(profiles_ds, namelist)
        else:
            print(f"No pi groups found in output. Computing offline...")
            pi_groups = compute_pi_groups(profiles_ds, timeseries_ds, reference_ds, namelist)
        ds = xr.merge([profiles_ds, pi_groups])
        pi_fig, pi_ax = plot_pi_entr(ds,
                entr_type = "total_rate",
                xlims = (- 0.001, 3e-2),
                # aux_field_vals = (timeseries_ds["updraft_cloud_base"].item(), timeseries_ds["updraft_cloud_top"].item()),
                save_fig_path = os.path.join(save_figs_dir, "entr_profiles", rel_path.split("/")[-1] + ".png"))
        profiles.append(ds)

        # plot profiles
        profiles_ds, timeseries_ds, reference_ds = preprocess_stats(stats, t_interval_from_end_s = T_INTERVAL_FROM_END,)
        les_path = namelist["meta"].get('lesfile', None)
        if les_path:
            les_path = full_les_path(les_path)
        if les_path:
            profiles_ds_les, _ , _ = preprocess_stats(les_path,
                                                    t_interval_from_end_s = T_INTERVAL_FROM_END,
                                                    interp_z = profiles_ds.zc,
                                                    rectify_surface_fluxes = True)

            profiles_ds_les_unaveraged, _ , _ = preprocess_stats(les_path, interp_z = profiles_ds.zc, t_interval_from_end_s = None)
            profiles_ds_les_std = compute_std_time(profiles_ds_les_unaveraged)
        
        else:
            profiles_ds_les = None
            profiles_ds_les_std = None


        prof_fig, prof_ax = plot_profiles(profiles_ds, profiles_ds_les, profiles_ds_les_std,
                    save_fig_path = os.path.join(save_figs_dir, "profiles", rel_path.split("/")[-1] + ".png"))

        prof_fig, prof_ax = plot_profiles(profiles_ds, plot_field_names = ["nh_pressure", "nh_pressure_adv", "nh_pressure_drag", "nh_pressure_b", "massflux_grad", "tke_mean"],
                    save_fig_path = os.path.join(save_figs_dir, "profiles_more", rel_path.split("/")[-1] + ".png"))

        if "tke_term_1" in profiles_ds:
            plot_profiles(profiles_ds, plot_field_names = TKE_TERMS, ylims = [0, 1000], nrows = 4, ncols = 4,
                        save_fig_path = os.path.join(save_figs_dir, "profiles_tke", rel_path.split("/")[-1] + ".png"))

            terms = ("eddy_diffusivity", "eddy_viscosity", "mixing_length", "tke_mean")
            plot_profiles(profiles_ds, plot_field_names = terms, ylims = [0, 1000], nrows = 4, ncols = 4,
                        save_fig_path = os.path.join(save_figs_dir, "profiles_other", rel_path.split("/")[-1] + ".png"))

            plot_profiles_one_plot(profiles_ds, reference_ds, plot_field_names = TKE_TERMS, ylims = [0, 1000], nrows = 4, ncols = 4,
                        save_fig_path = os.path.join(save_figs_dir, "profiles_tke_one_zoom", rel_path.split("/")[-1] + ".png"))

            plot_profiles_one_plot(profiles_ds, reference_ds, plot_field_names = TKE_TERMS, ylims = [0, 4000], nrows = 4, ncols = 4,
                        save_fig_path = os.path.join(save_figs_dir, "profiles_tke_one", rel_path.split("/")[-1] + ".png"))

        # plot timeseries
        profiles_ds, timeseries_ds, reference_ds = preprocess_stats(stats, t_interval_from_end_s = None)
    
        plot_timeseries(profiles_ds,
                        save_fig_path = os.path.join(save_figs_dir, "timeseries", rel_path.split("/")[-1] + ".png"))
        plot_timeseries(profiles_ds,
                        plot_field_names = TIMESERIES_UPDRAFT_DEFAULT_FIELDS,
                        save_fig_path = os.path.join(save_figs_dir, "timeseries_updraft", rel_path.split("/")[-1] + ".png"))

        plot_timeseries(profiles_ds,
                        filt_zero = True,
                        plot_field_names = ["env_qi", "env_tke", "env_buoyancy", "env_w", "env_ql", "env_cloud_fraction", "eddy_diffusivity", "eddy_viscosity", "env_ql", "env_temperature", "env_RH", "env_buoyancy", ],#"mixing_length"],
                        save_fig_path = os.path.join(save_figs_dir, "timeseries_env", rel_path.split("/")[-1] + ".png"))

        plot_timeseries(profiles_ds,
                        filt_zero = True,
                        cmap = plt.get_cmap("seismic"),
                        plot_field_names = BG_GRAD_FIELDS,
                        save_fig_path = os.path.join(save_figs_dir, "timeseries_bg", rel_path.split("/")[-1] + ".png"))

        plot_timeseries(profiles_ds, ylims = [0, 1000],
                        nrows = 4, ncols = 4, 
                        plot_field_names = TKE_TERMS,
                        filt_zero = False,
                        cmap = plt.get_cmap("seismic"),
                        save_fig_path = os.path.join(save_figs_dir, "timeseries_tke", rel_path.split("/")[-1] + ".png"))

        plot_timeseries(profiles_ds, ylims = [0, 4000],
                        nrows = 4, ncols = 4, 
                        plot_field_names = TKE_TERMS,
                        filt_zero = False,
                        cmap = plt.get_cmap("seismic"),
                        save_fig_path = os.path.join(save_figs_dir, "timeseries_tke_nonzoom", rel_path.split("/")[-1] + ".png"))
        if "entr_rate_inv_s" in profiles_ds:
            profiles_full.append(profiles_ds[ENTR_DETR_VARS])


    xr.concat(profiles, dim = "zc").to_netcdf(os.path.join(save_figs_dir, "profiles.nc"))
    xr.concat(profiles_full, dim = "zc").to_netcdf(os.path.join(save_figs_dir, "profiles_full.nc"))

if __name__ == '__main__':
    main()
