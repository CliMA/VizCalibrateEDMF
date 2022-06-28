import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import argparse
import json
import matplotlib.ticker as mticker
import matplotlib as mpl

###

# This code is designed to receive loss.nc from CalibrateEDMF's loss_map.jl function
# and plot a corresponding "corner plot" where the diagnoal is the marginal map per parameter
# averaged over all other parameters and the lower half of the matrix shows contour plots
# for all 2-parameter combinations in loss.nc
# to run from command line:
# python corner_plot.py PATH_TO_LOSS 'mean' -1

###

def main():
    parser = argparse.ArgumentParser(prog='corner_plot')
    parser.add_argument("file_path")
    parser.add_argument("ensamble_moment")
    parser.add_argument("case_number", type=int)
    args = parser.parse_args()
    file_path = args.file_path
    ensamble_moment = args.ensamble_moment
    case_number = args.case_number

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

    M = np.size(param_names)
    inx_mat = np.reshape(np.linspace(1,M*M,M*M),(M,M)).astype(int)
    X_inx_mat = (inx_mat-1)//4
    Y_inx_mat = (inx_mat-1)%4
    xl, yl = np.tril_indices_from(inx_mat, k=-1)

    group_names_rev = np.flipud(group_names)
    Z_matrix = ["" for x in range(M*M)]
    X_matrix = ["" for x in range(M*M)]
    Y_matrix = ["" for x in range(M*M)]
    for i in range(0,np.size(xl)):
        Z_matrix[inx_mat[xl[i],yl[i]]-1] = group_names_rev[i]
        y_, x_ = group_names_rev[i].split('.')
        X_matrix[inx_mat[xl[i],yl[i]]-1] = x_
        Y_matrix[inx_mat[xl[i],yl[i]]-1] = y_

    x_matrix = np.reshape(X_matrix, (M,M))
    y_matrix = np.reshape(Y_matrix, (M,M))
    z_matrix = np.reshape(Z_matrix, (M,M))
    y_, x_ = z_matrix[-1,0].split('.')
    z_matrix[M-1,M-1] = y_
    for k in range(0,M-1):
        y_, x_ = z_matrix[-1,k].split('.')
        z_matrix[k,k] = x_
        x_matrix[k,k] = z_matrix[k,k]
    x_matrix[-1,-1] = z_matrix[-1,-1]

    fig, axes = plt.subplots(M, M)
    max_y = 0.0
    for i in range(0,M):
        for j in range(0,M):
            labelx = name_dict.get(x_matrix[i,j])
            labely = name_dict.get(y_matrix[i,j])
            if i==j:
                # Marginal loss on the diagonal:
                # if on the diagonal: collect all fields that has this parameter,
                # take thier average with respect to the other parameter and plot
                # the marginal loss curve on log-scale by scanning vertically
                # and horiozntally from this diagonal spot.
                # Here for the variable in [1,1] collect [1,0] from the horizontal and [2,1], [3,1] from the veritcal axis
                # for the variable in [2,2] collect [2,0] [2,1] from the horizontal and [3,2] from the veritcal axis
                # generally for [k,k] collect from the horizontal axis [0:k-1] and from the veritcal axis [k,k+1:M]
                #   |---|---|---|---|      |---|---|---|---|
                #   |0,0|   |   |   |      |0,0|   |   |   |
                #   |---|---|---|---|      |---|---|---|---|
                #   | H |1,1|   |   |      |   |1,1|   |   |
                #   |---|---|---|---|  or  |---|---|---|---|
                #   |   | V |2,2|   |      | H | H |2,2|   |
                #   |---|---|---|---|      |---|---|---|---|
                #   |   | V |   |3,3|      |   |   | V |3,3|
                #   |---|---|---|---|      |---|---|---|---|
                for k in range(0,M): # scan horizontally
                    if k<j:
                        group_name = z_matrix[i,k]
                        # if at k=0 (first instant of this variable) create z_diag
                        if k==0:
                            x = np.array(data.groups[group_name].variables[x_matrix[i,j]])
                            z_diag = np.zeros_like(x)
                        z = np.squeeze(np.array(data.groups[group_name].variables["loss_data"])[0,0,:,:])
                        z_mean = np.nanmean(z, axis = 1)
                        z_diag = np.add(z_diag, z_mean)
                    elif k>i:
                        group_name = z_matrix[k,j]
                        # if at i=j=0 we never stepd into the previous condition create z_diag
                        if j==0:
                            x = np.array(data.groups[group_name].variables[x_matrix[i,j]])
                            z_diag = np.zeros_like(x)
                        z = np.squeeze(np.array(data.groups[group_name].variables["loss_data"])[0,0,:,:])
                        z_mean = np.nanmean(z, axis = 1)
                        z_diag = np.add(z_diag, z_mean)
                    else:
                        pass

                ax = axes[i][j]
                pt = ax.plot(x, z_diag)
                max_y = np.max((max_y,np.nanmax(z_diag)))
                if i==M-1:
                    ax.set_xlabel(labelx)
                else:
                    xlabels = [item.get_text() for item in ax.get_xticklabels()]
                    xempty_string_labels = [''] * len(xlabels)
                    ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks().tolist()))
                    ax.set_xticklabels(xempty_string_labels)
                ax.yaxis.tick_right()

            # if off diagonal, identify and load relevant 2D slice for contour map
            elif bool(z_matrix[i,j]):
                group_name = z_matrix[i,j]
                x = np.array(data.groups[group_name].variables[x_matrix[i,j]])
                y = np.array(data.groups[group_name].variables[y_matrix[i,j]])
                z = get_loss(np.array(data.groups[group_name].variables["loss_data"]), ensamble_moment, case_number)
                z = np.squeeze(np.array(data.groups[group_name].variables["loss_data"])[0,0,:,:])
                ax = axes[i][j]
                pcm = ax.contourf(x, y, np.fliplr(np.rot90(z, k=3)), cmap = "RdYlBu_r")
                if j==0 and i==M-1:
                    ax.set_xlabel(labelx)
                    ax.set_ylabel(labely)
                elif j==0:
                    xlabels = [item.get_text() for item in ax.get_xticklabels()]
                    xempty_string_labels = [''] * len(xlabels)
                    ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks().tolist()))
                    ax.set_xticklabels(xempty_string_labels)
                    ax.set_ylabel(labely)
                elif i==M-1:
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

            else:
                axes[i,j].set_visible(False)
            axes[i,j].tick_params(axis='both', labelsize=8)
    fig.colorbar(pcm, ax=axes, panchor = "NE", location='top', aspect=50, anchor = (1.0,0.3))
    for i in range(0,M):
        for j in range(0,M):
            if i==j:
                ax = axes[i][j]
                ax.set_ylim(0,max_y)
    plt.show()

def get_loss(z_full, ensamble_moment, case_number = -1):
    if ensamble_moment == "mean":
        z_ens = np.mean(z_full, axis = 0)
    elif ensamble_moment == "variance":
        z_ens = np.squeeze(np.var(z_full, axis = 0))
    else:
        print("ensamble_moment should be either mean or variance")
    if case_number == -1:
        z_case = np.squeeze(np.mean(z_ens), axis = 0)
    else:
        z_case = np.squeeze(z_ens[int(case_number)-1,:,:])
    return z_case

if __name__ == '__main__':
    main()
