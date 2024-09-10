import argparse
from edmf_viz.plot_profiles_helper import *
import os
from glob import glob
from matplotlib.lines import Line2D
from src.data_tools import *



T_INTERVAL_FROM_END = 12.0 * 3600.0

# path to TC.jl output root folder (or on Zenodo (see Christopoulos et al., 2024))
data_rel_path = "/central/groups/esm/cchristo/data/james_2024_submission/calibration_diagnostics"
linreg_entr_data_dir = os.path.join(data_rel_path, "linreg_full_cal/AMIP4K_simulations")
nn_entr_data_dir =  os.path.join(data_rel_path, "nn_full_cal/AMIP4K_simulations")


def make_entr_profs_for_dir(data_dir, save_figs_dir, **kwargs):

    print(f"Data directory: {data_dir}")
    print(f"Figures will be saved in: {save_figs_dir}")

    plt.rc('legend',fontsize=16)

    rel_paths = os.listdir(data_dir)

    for i, rel_path in enumerate(rel_paths):
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
                xlims = (0.0, 3e-2),
                save_fig_path = os.path.join(save_figs_dir, rel_path.split("/")[-1] + ".pdf"), **kwargs)

def main():

    make_entr_profs_for_dir(linreg_entr_data_dir, "./figs/linreg_entr_profiles", xlabel_bool = False, legend_bool = True, ylims = [0, 1750], tick_label_size = 14)
    make_entr_profs_for_dir(nn_entr_data_dir, "./figs/nn_entr_profiles", xlabel_bool = True, legend_bool = False, ylims = [0, 1750], tick_label_size = 14)


if __name__ == '__main__':
    main()
