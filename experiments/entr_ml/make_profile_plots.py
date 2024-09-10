import argparse
from edmf_viz.plot_profiles_helper import *
import os
from glob import glob
from matplotlib.lines import Line2D
from src.data_tools import *
import matplotlib.ticker as ticker



T_INTERVAL_FROM_END = 12.0 * 3600.0
NCOLS = 3
NROWS = 3

data_rel_path = "/central/groups/esm/cchristo/data/james_2024_submission" #

# path to TC.jl output
data_dir_linreg = os.path.join(data_rel_path, "calibration_diagnostics/linreg_full_cal/AMIP4K_simulations")
data_dir_nn = os.path.join(data_rel_path, "calibration_diagnostics/nn_full_cal/AMIP4K_simulations")
data_dir_md = os.path.join(data_rel_path, "baseline_EDMF_simulations/md_AMIP4K")

rel_paths = os.listdir(data_dir_linreg)

fig_profiles, axs_profiles = plt.subplots(3, NCOLS, sharey=True, figsize=(8, 10))

for i, rel_path in enumerate(rel_paths):
    rel_path_ = os.path.join(data_dir_linreg, rel_path)
    stats_linreg = stats_path(rel_path_)
    namelist = load_namelist(namelist_path(rel_path_))

    rel_path_ = os.path.join(data_dir_nn, rel_path)
    stats_nn = stats_path(rel_path_)

    rel_path_ = os.path.join(data_dir_md, rel_path.rsplit('.', 1)[0])
    stats2 = stats_path(rel_path_)

    #### plot profiles

    # process EDMF data
    profiles_ds_linreg, _, _ = preprocess_stats(stats_linreg, t_interval_from_end_s = T_INTERVAL_FROM_END,)
    profiles_ds_nn, _, _ = preprocess_stats(stats_nn, t_interval_from_end_s = T_INTERVAL_FROM_END,)
    profiles_md, timeseries_ds2, reference_ds2 = preprocess_stats(stats2, t_interval_from_end_s = T_INTERVAL_FROM_END,)
    les_path = namelist["meta"].get('lesfile', None)

    # process LES data
    les_path = full_les_path(les_path)
    profiles_ds_les, _ , _ = preprocess_stats(les_path,
                                            t_interval_from_end_s = T_INTERVAL_FROM_END,
                                            interp_z = profiles_ds_linreg.zc,
                                            rectify_surface_fluxes = True)

    profiles_ds_les_unaveraged, _ , _ = preprocess_stats(les_path, interp_z = profiles_ds_linreg.zc, t_interval_from_end_s = None)
    profiles_ds_les_std = compute_std_time(profiles_ds_les_unaveraged)




    plot_profiles_combined(profiles_ds_linreg, profiles_ds_nn, profiles_md, profiles_ds_les, profiles_ds_les_std, ylims = [0, 2000],
                plot_field_names = ["ql_mean", "total_flux_qt", "total_flux_s"],
                ds1_label = "EDMF-Linreg",
                ds2_label = "EDMF-NN",
                ds3_label = "Cohen et al., 2020",
                nrows = 1, ncols = NCOLS,
                ax = axs_profiles[i],)

profile_row_names = ["Stratocumulus", "Transition", "Cumulus"] 
for i, label in enumerate(profile_row_names):
    axs_profiles[i, 0].set_ylabel(label, weight = "bold")

profile_col_names = [r"$\mathbf{\bar{q_l}}$ $[kg \cdot kg^{-1}]$", r"$\mathbf{\overline{w'q_t'}}$ $[m \cdot s^{-1}]$", r"$\mathbf{\overline{w's'}}$ $[J \cdot m \cdot s^{-1} \cdot K^{-1}]$"]


for i, label in enumerate(profile_col_names):
    axs_profiles[-1, i].set_xlabel(label, weight = "bold", fontsize = 12, labelpad=13)

# Add legend
legend_labels = ["LES", "EDMF-Linreg", "EDMF-NN", "Cohen et al., 2020",]
legend_handles = [Line2D([0], [0], color='k', lw=2), Line2D([0], [0], color='r', lw=2), Line2D([0], [0], color='b', lw=2), Line2D([0], [0], color='gray', lw=2, linestyle='--')]
axs_profiles[0, -1].legend(legend_handles, legend_labels, loc='upper right', fontsize = "x-small", prop={'weight': 'bold'}, handlelength=0.8, handletextpad=0.3, borderpad=0.1)


fig_profiles.subplots_adjust(hspace=0.25, wspace = 0.1)

axs_profiles[0, 0].set_xlim([0.0, 6e-4])


class CustomScalarFormatter(ticker.ScalarFormatter):
    def _set_order_of_magnitude(self):
        """Override to customize the offset text format."""
        self.orderOfMagnitude = -4

    def get_offset(self):
        """Override to return the desired 1e-4 format."""
        return '1e-4'


formatter = CustomScalarFormatter(useMathText=True)
formatter.set_powerlimits((-4, -4))
axs_profiles[0, 0].xaxis.set_major_formatter(formatter)
axs_profiles[0, 0].xaxis.get_offset_text().set_position((1, 0.05))



axs_profiles[1, 0].set_xlim([1e-7, 5e-5])
axs_profiles[2, 0].set_xlim([1e-7, 7e-5])

fig_save_path = os.path.join("./figs/", "fig3_profile_grid.pdf")
fig_profiles.savefig(fig_save_path, dpi=200)

