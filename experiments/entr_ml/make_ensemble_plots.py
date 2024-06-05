import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from src.diag_plots import *
from src.data_tools import *

# path to CEDMF output Diagnostics.nc file
linreg_diag_path = "/groups/esm/cchristo/cedmf_results/james_v1_runs/results_Inversion_p22_e300_i15_mb_LES_2024-03-15_10-08_Vxo_longer_long_run/Diagnostics.nc"

ds_dict = preprocess_diags(linreg_diag_path)

plot_y_profiles_iter(ds_dict, 
                    iterations = [1, 50],
                    val_bool = True, 
                    save_fig_dir = "./figs/y_ensemble_profiles_linreg",
                    case_ind = 2,
                    nrows = 2, 
                    ncols = 6)


