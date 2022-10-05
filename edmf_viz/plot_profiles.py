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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tc_output_dir")
    parser.add_argument("--save_figs_dir")
    args = parser.parse_args()

    data_dir = args.tc_output_dir
    save_figs_dir = args.save_figs_dir

    profiles = []
    for rel_path in os.listdir(data_dir):
        rel_path = os.path.join(data_dir, rel_path)
        stats = stats_path(rel_path)
        namelist = load_namelist(namelist_path(rel_path))
        try:
            # plot entrainment/detrainment
            profiles_ds, timeseries_ds, reference_ds = preprocess_stats(stats, drop_zero_area = True)
            pi_groups = compute_pi_groups(profiles_ds, timeseries_ds, reference_ds, namelist)
            ds = xr.merge([profiles_ds, pi_groups])
            pi_fig, pi_ax = plot_pi_entr(ds,
                    aux_field_vals = (timeseries_ds["updraft_cloud_base"].item(), timeseries_ds["updraft_cloud_top"].item()),
                    save_fig_path = os.path.join(save_figs_dir, "entr_profiles", rel_path.split("/")[-1] + ".png"))
            profiles.append(ds)

            # plot profiles
            profiles_ds, timeseries_ds, reference_ds = preprocess_stats(stats)
            les_path = full_les_path(namelist['meta']['lesfile'])
            profiles_ds_les, _ , _ = preprocess_stats(les_path)
            profiles_ds_les_std = compute_std_time(les_path)
            prof_fig, prof_ax = plot_profiles(profiles_ds, profiles_ds_les, profiles_ds_les_std,
                        save_fig_path = os.path.join(save_figs_dir, "profiles", rel_path.split("/")[-1] + ".png"))
        except:
            print("Failed to plot. ", rel_path)
    xr.concat(profiles, dim = "zc").to_netcdf(os.path.join(save_figs_dir, "profiles.nc"))

if __name__ == '__main__':
    main()
