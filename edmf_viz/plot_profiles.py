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
    parser.add_argument("--aux_plots", required=False, type=bool, help="Whether to include auxiliary plots.")
    args = parser.parse_args()

    data_dir = args.tc_output_dir
    aux_plots = args.aux_plots

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
        try:
            # plot entrainment/detrainment
            profiles_ds, timeseries_ds, reference_ds = preprocess_stats(stats, t_interval_from_end_s = T_INTERVAL_FROM_END, drop_zero_area = True)
            pi_var_names = get_pi_var_names(namelist)
            # if pi groups saved in TC stats file, use those. Otherwise compute offline
            if any(pi_var_name in profiles_ds for pi_var_name in pi_var_names):
                pi_groups = get_pi_groups(profiles_ds, namelist)
            else:
                print(f"No pi groups found in output. Computing offline...")
                pi_groups = compute_pi_groups(profiles_ds, timeseries_ds, reference_ds, namelist)
            ds = xr.merge([profiles_ds, pi_groups])
            pi_fig, pi_ax = plot_pi_entr(ds,
                    aux_field_vals = (timeseries_ds["updraft_cloud_base"].item(), timeseries_ds["updraft_cloud_top"].item()),
                    save_fig_path = os.path.join(save_figs_dir, "entr_profiles", rel_path.split("/")[-1] + ".png"))
            profiles.append(ds)

            # plot profiles
            profiles_ds, timeseries_ds, reference_ds = preprocess_stats(stats, t_interval_from_end_s = T_INTERVAL_FROM_END,)
            les_path = full_les_path(namelist['meta']['lesfile'])
            profiles_ds_les, _ , _ = preprocess_stats(les_path,
                                                    t_interval_from_end_s = T_INTERVAL_FROM_END,
                                                    interp_z = profiles_ds.zc,
                                                    rectify_surface_fluxes = True)

            profiles_ds_les_unaveraged, _ , _ = preprocess_stats(les_path, interp_z = profiles_ds.zc, t_interval_from_end_s = None)
            profiles_ds_les_std = compute_std_time(profiles_ds_les_unaveraged)

            prof_fig, prof_ax = plot_profiles(profiles_ds, profiles_ds_les, profiles_ds_les_std,
                        save_fig_path = os.path.join(save_figs_dir, "profiles", rel_path.split("/")[-1] + ".png"))

            # plot timeseries
            profiles_ds, timeseries_ds, reference_ds = preprocess_stats(stats, t_interval_from_end_s = None)
            profiles_ds_les, _ , _ = preprocess_stats(les_path, t_interval_from_end_s = None)
            plot_timeseries(profiles_ds,
                            save_fig_path = os.path.join(save_figs_dir, "timeseries", rel_path.split("/")[-1] + ".png"))
            plot_timeseries(profiles_ds,
                            plot_field_names = TIMESERIES_UPDRAFT_DEFAULT_FIELDS,
                            save_fig_path = os.path.join(save_figs_dir, "timeseries_updraft", rel_path.split("/")[-1] + ".png"))
            plot_timeseries(profiles_ds_les,
                            plot_field_names =[FIELD_MAP[field] for field in TIMESERIES_MEAN_DEFAULT_FIELDS if "tke_mean" not in field],
                            save_fig_path = os.path.join(save_figs_dir, "timeseries_les", rel_path.split("/")[-1] + ".png"))

            profiles_full.append(profiles_ds[ENTR_DETR_VARS])

            if aux_plots:
                # plot histograms 
                plot_histogram(profiles_ds, save_fig_path = os.path.join(save_figs_dir, "histograms", rel_path.split("/")[-1] + ".png"))

                plot_histogram(profiles_ds, variable_names = ("pi_1", "pi_2", "pi_3", "pi_4", "pi_5", "pi_6"),
                                save_fig_path = os.path.join(save_figs_dir, "pi_histograms", rel_path.split("/")[-1] + ".png"))


                # plot scatter plots
                plot_scatter(profiles_ds, "updraft_area", save_fig_path = os.path.join(save_figs_dir, "scatter_plots", rel_path.split("/")[-1] + ".png"))

                # more scatter plots 
                plot_scatter(profiles_ds, "ln_massflux_grad", save_fig_path = os.path.join(save_figs_dir, "scatter_plots2", rel_path.split("/")[-1] + ".png")) 
                plot_scatter(profiles_ds, "pi_3", save_fig_path = os.path.join(save_figs_dir, "scatter_plots3", rel_path.split("/")[-1] + ".png")) 
                plot_scatter(profiles_ds, "pi_4", save_fig_path = os.path.join(save_figs_dir, "scatter_plots4", rel_path.split("/")[-1] + ".png")) 

                plot_scatter(profiles_ds, "pi_4", save_fig_path = os.path.join(save_figs_dir, "scatter_plots5", rel_path.split("/")[-1] + ".png")) 

        except Exception as e:
            print("Failed to plot. ", rel_path)
            print(e)
    xr.concat(profiles, dim = "zc").to_netcdf(os.path.join(save_figs_dir, "profiles.nc"))
    xr.concat(profiles_full, dim = "zc").to_netcdf(os.path.join(save_figs_dir, "profiles_full.nc"))

if __name__ == '__main__':
    main()
