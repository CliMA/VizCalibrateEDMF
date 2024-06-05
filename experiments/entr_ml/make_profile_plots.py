import argparse
from edmf_viz.plot_profiles_helper import *
import os
from glob import glob
from matplotlib.lines import Line2D
from src.data_tools import *


T_INTERVAL_FROM_END = 12.0 * 3600.0
NCOLS = 3
NROWS = 3

# path to TC.jl output
data_dir = "/groups/esm/cchristo/clima/CalibrateEDMF.jl/tools/results_Inversion_p22_e300_i15_mb_LES_2024-03-15_10-08_Vxo_longer_long_run_last_nn_particle_mean_on_all_amip4K_samples"
data_dir2 = "/groups/esm/cchristo/clima/TurbulenceConvection.jl/output_mf_baseline_james_v1_amip4K_samples"

rel_paths = os.listdir(data_dir)

fig_profiles, axs_profiles = plt.subplots(3, NCOLS, sharey=True, figsize=(8, 10))

for i, rel_path in enumerate(rel_paths):
    rel_path1 = os.path.join(data_dir, rel_path)
    stats = stats_path(rel_path1)
    namelist = load_namelist(namelist_path(rel_path1))


    rel_path2 = os.path.join(data_dir2, rel_path.rsplit('.', 1)[0])
    stats2 = stats_path(rel_path2)

    #### plot profiles

    # process EDMF data
    profiles_ds, timeseries_ds, reference_ds = preprocess_stats(stats, t_interval_from_end_s = T_INTERVAL_FROM_END,)
    profiles_ds2, timeseries_ds2, reference_ds2 = preprocess_stats(stats2, t_interval_from_end_s = T_INTERVAL_FROM_END,)
    les_path = namelist["meta"].get('lesfile', None)

    # process LES data
    les_path = full_les_path(les_path)
    profiles_ds_les, _ , _ = preprocess_stats(les_path,
                                            t_interval_from_end_s = T_INTERVAL_FROM_END,
                                            interp_z = profiles_ds.zc,
                                            rectify_surface_fluxes = True)

    profiles_ds_les_unaveraged, _ , _ = preprocess_stats(les_path, interp_z = profiles_ds.zc, t_interval_from_end_s = None)
    profiles_ds_les_std = compute_std_time(profiles_ds_les_unaveraged)




    plot_profiles_combined(profiles_ds, profiles_ds2, profiles_ds_les, profiles_ds_les_std, ylims = [0, 2000],
                plot_field_names = ["ql_mean", "total_flux_qt", "total_flux_s"],

                nrows = 1, ncols = NCOLS,
                ax = axs_profiles[i],)

profile_row_names = ["Stratocumulus", "Transition", "Cumulus"] 
for i, label in enumerate(profile_row_names):
    axs_profiles[i, 0].set_ylabel(label, weight = "bold")

profile_col_names = [r"$\mathbf{\bar{q_l}}$ $[kg \cdot kg^{-1}]$", r"$\mathbf{\overline{w'q_t'}}$ $[m \cdot s^{-1}]$", r"$\mathbf{\overline{w's'}}$ $[J \cdot m \cdot s^{-1} \cdot K^{-1}]$"]


for i, label in enumerate(profile_col_names):
    axs_profiles[-1, i].set_xlabel(label, weight = "bold", fontsize = 12, labelpad=13)

# Add legend
legend_labels = ["LES", "EDMF-Linreg", "Cohen et al., 2020",]
legend_handles = [Line2D([0], [0], color='k', lw=2), Line2D([0], [0], color='r', lw=2), Line2D([0], [0], color='gray', lw=2, linestyle='--')]
axs_profiles[0, -1].legend(legend_handles, legend_labels, loc='upper right', fontsize = "x-small", prop={'weight': 'bold'}, handlelength=1.0, handletextpad=0.4, borderpad=0.1)


fig_profiles.subplots_adjust(hspace=0.25, wspace = 0.1)

axs_profiles[1, 0].set_xlim([1e-7, 5e-5])
axs_profiles[2, 0].set_xlim([1e-7, 7e-5])

fig_save_path = os.path.join("./figs/", "fig3_profile_grid.pdf")
fig_profiles.savefig(fig_save_path, dpi=200)

