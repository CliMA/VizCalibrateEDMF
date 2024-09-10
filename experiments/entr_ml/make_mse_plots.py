import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import sys
import os

# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# path to top-level VizCalibrateEDMF dir
project_root = "/groups/esm/cchristo/clima/VizCalibrateEDMF"
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from src.diag_plots import *
from compute_mse_parallel import *


# the required data can be found on Zenodo (see Christopoulos et al., 2024) or in /groups/esm/cchristo/data/james_2024_submission on Caltech HPC.
data_rel_path = "/groups/esm/cchristo/data/james_2024_submission"

# load profiles from training set
nn_path = os.path.join(data_rel_path, "calibration_diagnostics/nn_full_cal/nn_full_cal_unpacked_profiles_training.nc")
linreg_path = os.path.join(data_rel_path, "calibration_diagnostics/linreg_full_cal/linreg_full_cal_unpacked_profiles_training.nc")
# (5, 176, 6, 300, 80) (iters, #cases, #num_vars, #ensemble, #vertical levels)
ds_nn = xr.open_dataset(nn_path).load()
ds_linreg = xr.open_dataset(linreg_path)

# load profiles from validation set
nn_path_val = os.path.join(data_rel_path, "calibration_diagnostics/nn_full_cal/nn_full_cal_unpacked_profiles_validation.nc")
linreg_path_val = os.path.join(data_rel_path, "calibration_diagnostics/linreg_full_cal/linreg_full_cal_unpacked_profiles_validation.nc")
ds_nn_val = xr.open_dataset(nn_path_val).load()
ds_linreg_val = xr.open_dataset(linreg_path_val)

# load ensemble particle data 
nn_diag_path = os.path.join(data_rel_path, "calibration_diagnostics/nn_full_cal/output/Diagnostics.nc")
ds_diagnostics_nn = xr.open_dataset(nn_diag_path, group = "particle_diags")
ds_ref = xr.open_dataset(nn_diag_path, group = "reference")

linreg_diag_path = os.path.join(data_rel_path, "calibration_diagnostics/linreg_full_cal/output/Diagnostics.nc")
ds_diagnostics_linreg = xr.open_dataset(linreg_diag_path, group = "particle_diags")
ds_ref = xr.open_dataset(linreg_diag_path, group = "reference")

y_les_path = os.path.join(data_rel_path, "calibration_diagnostics/les_y_full_cal/les_y_profiles_training.nc")
ds_y = xr.open_dataset(y_les_path).load()
ds_y.coords['case'] = ds_y.coords['case'] + 1

y_les_path_val = os.path.join(data_rel_path, "calibration_diagnostics/les_y_full_cal/les_y_profiles_validation.nc")
ds_y_val = xr.open_dataset(y_les_path_val).load()
ds_y_val.coords['case'] = ds_y_val.coords['case'] + 1


# compute mse for training set
max_plot_iteration = 50 
num_variables = 6

mse_ds_nn, mse_by_particle_ds_nn = compute_offline_mse_from_unpacked_profiles(ds_nn, ds_y, ds_diagnostics_nn,  num_iterations = max_plot_iteration, num_variables = num_variables)
mse_ds_linreg, mse_by_particle_ds_linreg = compute_offline_mse_from_unpacked_profiles(ds_linreg, ds_y, ds_diagnostics_linreg,  num_iterations = max_plot_iteration, num_variables = num_variables)


mse_list = [mse_ds_linreg.copy(), mse_ds_nn.copy()]
mse_by_particle_list = [mse_by_particle_ds_linreg.copy(), mse_by_particle_ds_nn.copy(),]



mse_ds_nn_val, mse_by_particle_ds_nn_val = compute_offline_mse_from_unpacked_profiles(ds_nn_val, ds_y_val, ds_diagnostics_nn,  num_iterations = max_plot_iteration, num_variables = num_variables, val_bool = True)
mse_ds_linreg_val, mse_by_particle_ds_linreg_val = compute_offline_mse_from_unpacked_profiles(ds_linreg_val, ds_y_val, ds_diagnostics_linreg,  num_iterations = max_plot_iteration, num_variables = num_variables, val_bool = True)


mse_list_val = [mse_ds_linreg_val.copy(), mse_ds_nn_val.copy(),]
mse_by_particle_list_val = [mse_by_particle_ds_linreg_val.copy(), mse_by_particle_ds_nn_val.copy()]



var_names = ["s_mean", "ql_mean", "qt_mean", "total_flux_qt", "total_flux_s", "lwp_mean"]

# compute rmse over baseline simulations in the training set
baseline_training_set_dir = "/central/groups/esm/cchristo/cedmf_results/james_v1_runs/md_edmf_amip"
baseline_training_rmses = compute_rmse_from_dir(baseline_training_set_dir, var_names)

# compute rmse over baseline simulations in the validation set
baseline_validation_set_dir = os.path.join(data_rel_path, "baseline_EDMF_simulations/md_AMIP4K")
baseline_validation_rmses = compute_rmse_from_dir(baseline_validation_set_dir, var_names)

plot_labels = [
         "NN", 
         "Linreg",
        ]


ylims = {"s_mean": [0, 15],  "ql_mean": [0, 1e-4], "qt_mean": [0, 3e-3],
         "total_flux_qt": [0, 2e-5], "total_flux_s": [0, 1e-1], "lwp_mean": [0, 2e-3]}
res = plot_var_offline(mse_list, mse_by_particle_list, 
                        nrows = 6, 
                        ncols = 1,
                        var_names = var_names, 
                        box_and_whiskers = False,
                        iterations_per_epoch = 11,
                        ylims = ylims,
                        max_min_shading = True,
                        suptitle = None,
                        plot_name_map = PLOT_NAME_MAP_ABBREVIATED,
                        baseline_hz_line = baseline_training_rmses,
                        legend_labels = None,
                        curve_colors = ['red', 'blue'],
                        linewidth=2, 
                        save_fig_path = "./figs/mse_plots_train_v2_fig1.pdf",
                        plot_labels = plot_labels)


res = plot_var_offline(mse_list_val, mse_by_particle_list_val, 
                        nrows = 6, 
                        ncols = 1,
                        var_names = var_names, 
                        box_and_whiskers = False,
                        iterations_per_epoch = 11,
                        ylims = ylims,
                        max_min_shading = True,
                        ylabel_bool = False,
                        suptitle = None,
                        plot_name_map = PLOT_NAME_MAP_ABBREVIATED,
                        baseline_hz_line = baseline_validation_rmses,
                        legend_labels = ["EDMF-Linreg", "EDMF-NN", "Cohen et al., 2020",],
                        curve_colors = ['red', 'blue'],
                        linewidth=2.5,
                        save_fig_path = "./figs/mse_plots_val_v2_fig1.pdf",
                        plot_labels = plot_labels)




###### print latex mse tables 
"""Round mse values in dictionary to 3 sig figs."""
def round_dict(mse_dict):
        return {key: np.format_float_positional(value, precision=3, unique=False, fractional = False, trim='k') for key, value in mse_dict.items()}



### mse tables for training set
baseline_training_rmses_round =  round_dict(baseline_training_rmses) 

linreg_training_rmses = dict(zip(var_names, mse_list[0].isel(iteration = -1)))
linreg_training_rmses_round = round_dict(linreg_training_rmses)

nn_training_rmses = dict(zip(var_names, mse_list[1].isel(iteration = -1)))
nn_training_rmses_round = round_dict(nn_training_rmses)

table_headers = [PLOT_NAME_MAP_SYMBOLS[var] for var in var_names]

models = [nn_training_rmses_round, linreg_training_rmses_round, baseline_training_rmses_round]
model_names = ["EDMF-NN", "EDMF-Linreg", "Cohen et al., 2020",]
# Start building the LaTeX table
latex_table = "\\begin{table}[!ht] \n\\centering\n\\begin{tabular}{|l|" + "c|" * len(var_names) + "}\n\\hline\n"
latex_table += "\\textbf{EDMF Version - AMIP} & " + " & ".join(table_headers) + " \\\\\n\\hline\n"

for i, model in enumerate(models):
    row_values = [f"{float(model.get(var, '0')):.2e}" for var in var_names]
    latex_table += f"{model_names[i]} & " + " & ".join(row_values) + " \\\\\n"
    latex_table += "\\hline\n"

latex_table += "\\end{tabular}\n\\caption{Table of root mean squared error for EDMF variants. Reported rmse values for EDMF-NN and EDMF-Linreg are the ensemble-averaged rmse in final iteration.}\n\\label{tab:amip_rmse_comparison}\n\\end{table}"

print("------ Training rmse table -----------")

print(latex_table)


### mse tables for validation set

baseline_validation_rmses_round = round_dict(baseline_validation_rmses)

linreg_validation_rmses = dict(zip(var_names, mse_list_val[0].isel(iteration = -1)))
linreg_validation_rmses_round = round_dict(linreg_validation_rmses)

nn_validation_rmses = dict(zip(var_names, mse_list_val[1].isel(iteration = -1)))
nn_validation_rmses_round = round_dict(nn_validation_rmses)

table_headers = [PLOT_NAME_MAP_SYMBOLS[var] for var in var_names]

models = [nn_validation_rmses_round, linreg_validation_rmses_round, baseline_validation_rmses_round]
model_names = ["EDMF-NN", "EDMF-Linreg", "Cohen et al., 2020",]
# Start building the LaTeX table
latex_table = "\\begin{table}[!ht] \n\\centering\n\\begin{tabular}{|l|" + "c|" * len(var_names) + "}\n\\hline\n"
latex_table += "\\textbf{EDMF Version - AMIP4K} & " + " & ".join(table_headers) + " \\\\\n\\hline\n"

for i, model in enumerate(models):
    row_values = [f"{float(model.get(var, '0')):.2e}" for var in var_names]
    latex_table += f"{model_names[i]} & " + " & ".join(row_values) + " \\\\\n"
    latex_table += "\\hline\n"

latex_table += "\\end{tabular}\n\\caption{Root mean squared error for EDMF variants on AMIP4k validation set.}\n\\label{tab:amip4k_rmse_comparison}\n\\end{table}"

print("------ Validation rmse table -----------")

print(latex_table)