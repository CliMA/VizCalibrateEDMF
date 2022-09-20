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

def full_les_path(path):
    """Get absolute path to LES file, given path of the form `../../../../../zhaoyi/GCMForcedLES/`"""
    path_split = path.split("/")
    rel_path = os.path.join(*path_split[path_split.index("GCMForcedLES"):])
    return os.path.join("/central/groups/esm/zhaoyi", rel_path)
