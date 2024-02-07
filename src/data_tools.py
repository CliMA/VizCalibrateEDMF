# -*- coding: utf-8 -*-
# Ignacio Lopez Gomez, Costa Christopoulos
# November 2021
# Post-processing tools for netCDF4 datasets.

from typing import Tuple, List
from glob import glob
import json
import netCDF4 as nc
import numpy as np
import os
import xarray as xr
import seaborn
import re

def vectorize(data_path:str) -> Tuple[List[float], List[float]]:
    """Take two columns of csv format data and return as two arrays.

    Args:
     - data_path: Path to csv data.

    Returns:
     - Two arrays representing the two-column data for abscissas and ordinates.
    """
    data = open(data_path, 'r')
    aux = data.readlines()
    x = []; y = [] ;
    for line in aux:
        linelist = line.split(',')
        x.append(float(linelist[0]))
        y.append(float(linelist[1]))   
    return x,y

def preprocess_diags(ds_path):
    """Open all groups in calibration `Diagnostics.nc` file and return as dict.

    Args:
     - ds_path: path to Diagnostics.nc file
    """
    group_names = ("ensemble_diags", "reference", "particle_diags", "metrics", "prior")
    ds_dict = {}
    for group_name in group_names:
        ds_dict[group_name] = xr.open_dataset(ds_path, group = group_name)
        if not (group_name in ("reference", "prior")):
            ds_dict[group_name] = ds_dict[group_name].sel(iteration = slice(1, ds_dict[group_name]["iteration"].max().item() - 1))

    var_names = ds_dict["reference"]["ref_variable_names"].values[:,1]
    for var_name_i in range(len(var_names)):
        var_name = var_names[var_name_i]

        if "mse_by_var_full" in  ds_dict["particle_diags"]:
            ds_dict["particle_diags"]["rmse_{}_full".format(var_name)] = ds_dict["particle_diags"]["mse_by_var_full"].isel(config_field = var_name_i)
        elif  "rmse_by_var_full" in  ds_dict["particle_diags"]:
            ds_dict["particle_diags"]["rmse_{}_full".format(var_name)] = ds_dict["particle_diags"]["rmse_by_var_full"].isel(config_field = var_name_i)

        if "val_mse_by_var_full" in ds_dict["particle_diags"]:
            ds_dict["particle_diags"]["val_mse_{}_full".format(var_name)] = ds_dict["particle_diags"]["val_mse_by_var_full"].isel(config_field = var_name_i)
        elif "val_rmse_by_var_full" in ds_dict["particle_diags"]:
            ds_dict["particle_diags"]["val_rmse_{}_full".format(var_name)] = ds_dict["particle_diags"]["val_rmse_by_var_full"].isel(config_field = var_name_i)

    return ds_dict


def ncFetch(directory:str, group:str, variable:str) -> np.ndarray:
    """Reads a variable from a netCDF file.

    Args:
     - directory: Output directory containing a Diagnostics.nc file.
     - group: Dataset group to which the variable belongs.
     - variable: Name of the variable to be retrieved.

    Returns:
     - The requested variable as a numpy array.
    """
    f = os.path.join(directory, 'Diagnostics.nc')
    with nc.Dataset(f,'r') as ds: 
        return np.array(ds[group][variable])

def ncFetchDim(directory:str, group:str, dimension:str) -> int:
    """Reads a dimension from a netCDF file.

    Args:
     - directory: Output directory containing a Diagnostics.nc file.
     - group: Dataset group to which the variable belongs.
     - dimension: Name of the variable to be retrieved.

    Returns:
     - The requested dimension.
    """
    f = os.path.join(directory, 'Diagnostics.nc')
    with nc.Dataset(f,'r') as ds:
        return ds[group].dimensions[dimension].size

def load_namelist(namelist_path:str):
    """Given path to `namelist.in` file, load and return as a nested dictionary."""
    with open(namelist_path, 'r') as f:
        namelist = json.load(f)
    return namelist

def namelist_path(output_dir):
    """Given directory containing TC results, return path to `*.in` file."""
    file_paths = [os.path.join(output_dir, y) for x in os.walk(output_dir) for y in glob(os.path.join(x[0], '*.in'))]
    if len(file_paths) > 1:
        raise Exception("Multiple *.in files found in directory.")
    return file_paths[0]

def stats_path(output_dir, multi_path = False):
    """Given directory containing TC results, return path(s) to `*.nc` file(s)."""
    file_paths = [os.path.join(output_dir, y) for x in os.walk(output_dir) for y in glob(os.path.join(x[0], '*.nc'))]
    if multi_path:
        return file_paths
    else:
        if len(file_paths) > 1:
            raise Exception("Multiple *.nc files found in directory.")
        return file_paths[0]

def diagnostics_nc_path(rel_path):
    return os.path.join(rel_path, "Diagnostics.nc")

def full_les_path(path):
    """Get absolute path to LES file, given path of the form `../../../../../zhaoyi/GCMForcedLES/`"""
    path_split = path.split("/")
    rel_path = os.path.join(*path_split[path_split.index("GCMForcedLES"):])
    return os.path.join("/central/groups/esm/zhaoyi", rel_path)


def get_cfsite_les_path(
    cfsite_number,
    forcing_model = "HadGEM2-A",
    month = 7,
    experiment = "amip"):

    cfsite_number = str(cfsite_number)
    month = str(month).zfill(2)
    root_dir = r"/central/groups/esm/zhaoyi/GCMForcedLES/cfsite/{month}/{forcing_model}/{experiment}/".format(month = month, forcing_model = forcing_model, experiment = experiment)
    rel_dir = "_".join([r"Output.cfsite{cfsite_number}".format(cfsite_number = cfsite_number), forcing_model, experiment, r"2004-2008.{month}.4x".format(month = month)])
    dir_prefix = os.path.join(root_dir, rel_dir)
    fname = "Stats.cfsite{cfsite_number}_{forcing_model}_{experiment}_2004-2008.{month}.nc".format(cfsite_number = cfsite_number, forcing_model = forcing_model, experiment = experiment, month = month)
    return os.path.join(dir_prefix, "stats", fname)


import re

def get_cfsite_info_from_path(input_string):
    # Correct the pattern to accurately match the structure
    pattern = r"cfsite/(\d+)/([^/]+)/([^/]+)/.*cfsite(\d+)_([^_]+)_([^_]+)_.*\.(\d{2})\..*nc"
    
    match = re.search(pattern, input_string)
    
    if match:
        return {
            "forcing_model": match.group(2),
            "cfsite_number": match.group(1),
            "month": match.group(7),
            "experiment": match.group(3)
        }
    else:
        return {}


def get_shallow_LES_library():
    # Initialize the LES library with model keys and empty dictionaries
    LES_library = {
        "HadGEM2-A": {},
        "CNRM-CM5": {},
        "CNRM-CM6-1": {}
    }
    
    # Define Shen_et_al_sites using a list comprehension
    Shen_et_al_sites = list(range(4, 16)) + list(range(17, 24))
    
    # A helper function to compute set differences, more concise than repeating setdiff
    def set_diff(base, exclude):
        return [item for item in base if item not in exclude]
    
    # A template for initializing model data to avoid repetition
    def init_model_data(model, sites_diffs):
        for month in ["01", "04", "07", "10"]:
            exclusions = sites_diffs.get(month, [])
            LES_library[model][month] = {
                "cfsite_numbers": set_diff(Shen_et_al_sites, exclusions),
                "experiments": ["amip", "amip4K"]
            }
    
    # Initialize data for each model with specific site differences
    init_model_data("HadGEM2-A", {
        "04": [15, 17, 18],
        "01": [15, 17, 18, 19, 20]
    })
    init_model_data("CNRM-CM5", {
        "10": [15, 22, 23],
        "07": [13, 14, 15, 18],
        "04": [11, 12, 13, 14, 15, 17, 18, 21, 22, 23],
        "01": [14, 15, 17, 18, 19, 20, 21, 22, 23]
    })
    init_model_data("CNRM-CM6-1", {
        "10": [22, 23],
        "07": [12, 13, 14, 15, 17],
        "04": [13, 14, 15],
        "01": [14, 15, 21, 22, 23]
    })
    
    return LES_library


def generate_all_les_paths(LES_library):
    paths = []
    # Iterate through each model in the library
    for model in LES_library:
        for month in LES_library[model]:
            experiments = LES_library[model][month].get('experiments', [])
            cfsite_numbers = LES_library[model][month].get('cfsite_numbers', [])
            for experiment in experiments:
                for cfsite_number in cfsite_numbers:
                    path = get_cfsite_les_path(cfsite_number,
                                                forcing_model = model,
                                                month = month,
                                                experiment = experiment)
                    paths.append(path)
    return paths
