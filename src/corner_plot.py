import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import json
import matplotlib.ticker as mticker

# command line:
# python corner_plot.py '/Users/yaircohen/Documents/codes/CliMA/calibrateEDMF_output/grid_search/Mar29/loss_hypercube.nc'
def main():
    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("file_path")
    args = parser.parse_args()
    file_path = args.file_path

    symbols_file = open('symbols.txt').read()
    name_dict = json.loads(symbols_file)

    data = nc.Dataset(file_path, 'r')
    param1 = []
    param2 = []
    group_names = []
    for group in data.groups:
        group_names.append(group)
        param1_, param2_ = group.split('.')
        param1.append(param1_)
        param2.append(param2_)
    param_names = list(set(param1 + param2))
    group_names = np.flipud(group_names)
    param1 = np.flipud(param1)
    param2 = np.flipud(param2)

    M = np.size(param_names)
    inx_mat = np.reshape(np.linspace(1,M*M,M*M),(M,M)).astype(int)
    xl, yl = np.tril_indices_from(inx_mat, k=-1)
    fig = plt.figure('loss corner plot')

    for i in range(0,np.size(xl)):
        x = np.array(data.groups[group_names[i]].variables[param2[i]])
        y = np.array(data.groups[group_names[i]].variables[param1[i]])
        ax = fig.add_subplot(M, M, inx_mat[xl[i],yl[i]])
        loss = np.squeeze(np.clip(np.array(data.groups[group_names[i]].variables["loss_data"]), -10, 10)[0,2,:,:])
        im = ax.contourf(x, y, np.fliplr(np.rot90(loss, k=3)), cmap = "RdYlBu_r")
        fig.colorbar(im)
        labelx = name_dict.get(param2[i])
        labely = name_dict.get(param1[i])
        if (any(inx_mat[:,0] == inx_mat[xl[i],yl[i]]) and any(inx_mat[-1,:] == inx_mat[xl[i],yl[i]])):
            ax.set_xlabel(labelx)
            ax.set_ylabel(labely)
        elif any(inx_mat[:,0] == inx_mat[xl[i],yl[i]]):
            xlabels = [item.get_text() for item in ax.get_xticklabels()]
            xempty_string_labels = [''] * len(xlabels)
            ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks().tolist()))
            ax.set_xticklabels(xempty_string_labels)
            ax.set_ylabel(labely)
        elif any(inx_mat[-1,:] == inx_mat[xl[i],yl[i]]):
            ylabels = [item.get_text() for item in ax.get_yticklabels()]
            yempty_string_labels = [''] * len(ylabels)
            ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks().tolist()))
            ax.set_yticklabels(yempty_string_labels)
            ax.set_xlabel(labelx)
        else:
            xlabels = [item.get_text() for item in ax.get_xticklabels()]
            xempty_string_labels = [''] * len(xlabels)
            ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks().tolist()))
            ax.set_xticklabels(xempty_string_labels)
            ylabels = [item.get_text() for item in ax.get_yticklabels()]
            yempty_string_labels = [''] * len(ylabels)
            ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks().tolist()))
            ax.set_yticklabels(yempty_string_labels)

    plt.show()

if __name__ == '__main__':
    main()