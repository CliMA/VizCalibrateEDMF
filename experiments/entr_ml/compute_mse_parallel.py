import argparse
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from edmf_viz.plot_profiles_helper import *
from src.data_tools import *

T_INTERVAL_FROM_END = 12.0 * 3600.0

def compute_mse_i(rel_path, T_INTERVAL_FROM_END, var_names):
    stats = stats_path(rel_path)
    namelist = load_namelist(namelist_path(rel_path))
    les_info = get_cfsite_info_from_path(namelist["meta"]["lesfile"])
    
    profiles_ds, timeseries_ds, reference_ds = preprocess_stats(stats, t_interval_from_end_s=T_INTERVAL_FROM_END, var_names=[field for field in var_names])


    les_path = full_les_path(namelist['meta']['lesfile'])
    profiles_ds_les, timeseries_ds_les, _ = preprocess_stats(les_path,
                                             t_interval_from_end_s=T_INTERVAL_FROM_END,
                                             interp_z=profiles_ds.zc,
                                             var_names=[FIELD_MAP[field] for field in var_names],
                                             rectify_surface_fluxes=True)

    mse_by_var = {}
    n_by_var = {}
    for var_name in var_names:
        if var_name != "lwp_mean":
            mse_by_var[var_name] =  np.sum((profiles_ds[var_name] - profiles_ds_les[FIELD_MAP[var_name]])**2).item()
            n_by_var[var_name] = len(profiles_ds[var_name])
        else:
            mse_by_var[var_name] =  (timeseries_ds[var_name] - timeseries_ds_les[FIELD_MAP[var_name]]**2).item()
            n_by_var[var_name] = 1
    return mse_by_var, n_by_var

def compute_rmse_from_dir(data_dir, var_names):

    print(f"Computing rmse by var for data directory: {data_dir}")
    rel_paths = [os.path.join(data_dir, rel_path) for rel_path in os.listdir(data_dir)]
    total_mse_by_var = {var_name: 0.0 for var_name in var_names}
    total_n_by_var = {var_name: 0.0 for var_name in var_names}
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_mse_i, rel_path, T_INTERVAL_FROM_END, var_names) for rel_path in rel_paths]

        for future in futures:
            mse_by_var, n_by_var = future.result()
            for var_name in var_names:
                total_mse_by_var[var_name] += mse_by_var[var_name]
                total_n_by_var[var_name] += n_by_var[var_name]

    rmse_by_var_out = {var: np.sqrt(total_mse_by_var[var] / total_n_by_var[var]) for var in var_names}
    print(rmse_by_var_out)
    return rmse_by_var_out
