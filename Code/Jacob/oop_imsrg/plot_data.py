import matplotlib.pyplot as plt
import numpy as np
import time
import os
import glob
import pickle

def plot_data(log_dir, plot_dir):

    data_files = glob.glob(log_dir+"*.pickle")

    pb_list = []
    for file in data_files:
        with open(file, 'rb') as f:
            pb_list.append(pickle.load(f))

    data = pb_list[-1]

    fig, axs = plt.subplots(2,2)
    axs[1,1].axis("off")
    fig.set_size_inches(10, 8)

    gvs = np.array([])
    pbs = np.array([])
    cov = np.array([])
    tms = np.array([])
    for lst in pb_list:
        gvs = np.append(gvs,lst[1:,3])
        pbs = np.append(pbs,lst[1:,4])
        cov = np.append(cov,lst[1:,0])
        tms = np.append(tms.astype(np.float),lst[1:,8])

    # PLOT 1 -- time to converge (or diverge) energy
    n = np.arange(len(tms))

    axs[0,0].plot(n, tms, marker='s')
    axs[0,0].set_xlabel('run number')
    axs[0,0].set_ylabel('time to convergence')
    axs[0,0].set_title(('Time to convergence \n'
                      '(d={:2.4f},g={:2.4f})').format(data[1,2],
                                                      data[1,3]))

    # PLOT 2 -- scatter plot that tracks which pb strength converged
    #           for the each value of g

    # pbs = data[1:, 4]
    # cov = data[1:, 0]
    # n = np.arange(len(times))

    axs[1,0].scatter(gvs, pbs, c=cov)
    axs[1,0].set_xlabel('g strength')
    axs[1,0].set_ylabel('pb strength')
    axs[1,0].set_title(('convergence space \n'
                      'for (d={:2.4f},g={:2.4f})').format(data[1,2],
                                                          data[1,3]))


    if os.path.isfile(log_dir+'total_mem.txt'):
        mem_vals = np.loadtxt(log_dir+'total_mem.txt', ndmin=1)
        n = np.arange(len(mem_vals))

        axs[0,1].plot(n, mem_vals, marker='s')
        axs[0,1].set_xlabel('run number')
        axs[0,1].set_ylabel('total memory allocation (in KiB)')
        axs[0,1].set_title('Running total memory allocation')


    plt.tight_layout() # resize so titles and axis labels fit

    # Save the plot
    current_time = time.localtime()
    fig.savefig(plot_dir+
        ('{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}'
            '_g-{:2.4f}.png').format(current_time[0], current_time[1],
                                     current_time[2], current_time[3],
                                     current_time[4], current_time[5],
                                     data[1,3]))
