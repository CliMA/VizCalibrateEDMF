# -*- coding: utf-8 -*-
# Ignacio Lopez Gomez
# November 2021
# Post-processing tools for netCDF4 datasets.

from typing import Tuple, List
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
