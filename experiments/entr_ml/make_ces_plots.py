import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np
import os
from matplotlib.font_manager import FontProperties


font_props = FontProperties(weight='bold', size=15)

# the required data can be found on Zenodo (see Christopoulos et al., 2024) or in /groups/esm/cchristo/data/james_2024_submission on Caltech HPC.
data_rel_path = "/groups/esm/cchristo/data/james_2024_submission"
plot_data_path = os.path.join(data_rel_path, "calibration_diagnostics/CES/CES_mcmc_samples.nc")

ds = xr.open_dataset(plot_data_path)

phi_optimal = ds["phi_optimal"].values
param_names = ds["param_names"].values
phi_optimal = [phi_optimal[np.where(param_names == "linear_ent_params_{{{}}}".format(i))[0][0]] for i in range(1, 13)]
phi_optimal = np.array(phi_optimal)

# final linreg weights, rounded to 1 sig fig
phi_optimal_full = np.loadtxt(os.path.join(data_rel_path, "calibration_diagnostics/CES/linreg_phi_optimal_full.txt")) 

# Compute/print the percent change
percent_change = (phi_optimal_full - phi_optimal) / phi_optimal * 100
print(percent_change)


# Set up the figure and subplots
fig, axes = plt.subplots(6, 2, figsize=(15, 20))

var_name_map = {
    "prior_linear_ent_params_{1}": r'$\mathbf{C^{\epsilon}_{1}}$',
    "prior_linear_ent_params_{2}": r'$\mathbf{C^{\epsilon}_{2}}$',
    "prior_linear_ent_params_{3}": r'$\mathbf{C^{\epsilon}_{3}}$',
    "prior_linear_ent_params_{4}": r'$\mathbf{C^{\epsilon}_{4}}$',
    "prior_linear_ent_params_{5}": r'$\mathbf{C^{\epsilon}_{5}}$',
    "prior_linear_ent_params_{7}": r'$\mathbf{C^{\delta}_{1}}$',
    "prior_linear_ent_params_{8}": r'$\mathbf{C^{\delta}_{2}}$',
    "prior_linear_ent_params_{9}": r'$\mathbf{C^{\delta}_{3}}$',
    "prior_linear_ent_params_{10}": r'$\mathbf{C^{\delta}_{4}}$',
    "prior_linear_ent_params_{11}": r'$\mathbf{C^{\delta}_{5}}$',
    "prior_linear_ent_params_{6}": r'$\mathbf{bias^{\epsilon}}$',
    "prior_linear_ent_params_{12}": r'$\mathbf{bias^{\delta}}$',
}

plot_vars = ["prior_linear_ent_params_{{{}}}".format(i) for i in range(1, 13)]
# Loop over the variables and plot each one
for i, var in enumerate(plot_vars):
    row = i % 6
    col = i // 6
    ax = axes[row, col]
    
    prior_data = ds[var].values
    posterior_data = ds[var.replace('prior', 'posterior')].values
    
    sns.histplot(prior_data, ax=ax, label='Precal Prior', color='blue', stat='density', kde=False, bins=50, alpha=0.4)
    sns.histplot(posterior_data, ax=ax, label='Precal Posterior', color='red', stat='density', kde=False, bins=50, alpha=0.4)

    ax.axvline(phi_optimal[i], color='red', label='Precal', linewidth=2)
    ax.axvline(phi_optimal_full[i], color='gray', linestyle='--', label='Full cal', linewidth=3)
    

    plot_label_name = var_name_map.get(var, var)

    ax.set_xlabel(f'{plot_label_name}', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=15)
    if col == 0:
        ax.set_ylabel('Density', fontsize=15, fontweight='bold')  # Add ylabels with custom font size
    else:
        ax.set_ylabel('')
    

    if (row == 0) & (col == 1):
        ax.legend(prop=font_props)

# Adjust layout
plt.tight_layout()
plt.show()

fig_fname = plot_data_path.split("/")[-1].split(".")[0]
plt.savefig(fig_fname + ".png")