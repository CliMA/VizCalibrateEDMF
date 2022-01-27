from typing import List
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib import colors
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from string import ascii_lowercase
from math import floor 

def get_tab_colors() -> List[str]:
    """Returns a list of tab colors."""
    return ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def convert2rgb(color, alpha=0.5):
    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except:
        c = color
    bg_rgb = (1, 1, 1)
    rgb = mc.to_rgb(c)
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]

def make_rgb_transparent(rgb, bg_rgb, alpha):
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]
    
font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 14}
rc('font',**font)
rc('text', usetex=False)

def interp_padeops_data(padeops_data, padeops_z, padeops_t, z_scm, t_scm):
    from scipy.interpolate import interp2d
    f_interp = interp2d(padeops_z, padeops_t, padeops_data)
    interp_data = f_interp(z_scm, t_scm)
    return interp_data

def label_axes(fig, labels=None, loc=None, **kwargs):
    from itertools import cycle
    from six.moves import zip
    from string import ascii_lowercase
    """
    Walks through axes and labels each.

    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure
         Figure object to work on

    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if labels is None:
        labels = ascii_lowercase

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (.9, .9)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(lab, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)

