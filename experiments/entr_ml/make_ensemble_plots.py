import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from src.diag_plots import *
from src.data_tools import *


# the required data can be found on Zenodo (see Christopoulos et al., 2024) or in /groups/esm/cchristo/data/james_2024_submission on Caltech HPC.
# path to CEDMF output Diagnostics.nc file
linreg_diag_path = "/central/groups/esm/cchristo/data/james_2024_submission/calibration_diagnostics/linreg_full_cal/output/Diagnostics.nc"

ds_dict = preprocess_diags(linreg_diag_path)

plot_y_profiles_iter(ds_dict, 
                    iterations = [1, 50],
                    val_bool = True, 
                    save_fig_dir = "./figs/y_ensemble_profiles_linreg",
                    case_ind = 2,
                    nrows = 2, 
                    ncols = 6)


