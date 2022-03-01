import argparse
from plot_entr import *
import os
from glob import glob
from src.data_tools import *

"""A command line interface to plot vertically-resolved pi groups and entrainment/detrainment fields, 
    given a path to completed TurbulenceConvection runs. Horizontal dashed lines indicate updraft cloud base and updraft cloud top.
    Shading indicates grid-mean cloud fraction. Pi groups are saved in a `.nc` file. 
    
    Example usage:
        python plot_entr_script.py data_dir=DATA_DIR save_figs_dir=SAVE_FIGS_DIR
    where DATA_DIR is a directory containing TurbulenceConvection runs, and SAVE_FIGS_DIR is the directory to save the resulting figures.
    """

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--save_figs_dir")
    args = parser.parse_args()

    data_dir = args.data_dir
    save_figs_dir = args.save_figs_dir

    pi_group_das = []
    for rel_path in os.listdir(data_dir):
        rel_path = os.path.join(data_dir, rel_path)
        stats = stats_path(rel_path)
        namelist = load_namelist(namelist_path(rel_path))
        profiles_ds, timeseries_ds, reference_ds = preprocess_stats(stats, drop_zero_area = True)
        pi_groups = compute_pi_groups(profiles_ds, timeseries_ds, reference_ds, namelist)
        ds = xr.merge([profiles_ds, pi_groups])
        plot_pi_entr(ds, 
                aux_field_vals = (timeseries_ds["updraft_cloud_base"].item(), timeseries_ds["updraft_cloud_top"].item()),
                save_file_name= os.path.join(save_figs_dir, rel_path.split("/")[-1] + ".png"))
        pi_group_das.append(pi_groups)

    xr.concat(pi_group_das, dim = "zc").to_netcdf(os.path.join(save_figs_dir, "pi_groups.nc"))

if __name__ == '__main__':
    main()
