#!/usr/bin/env python

import pdb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_check(xyz_1, xyz_2, xyz_t):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xyz_1[:,0], xyz_1[:,1], xyz_1[:,2], c='r', marker='o')
    ax.scatter(xyz_2[:,0], xyz_2[:,1], xyz_2[:,2], c='g', marker='o')
    ax.scatter(xyz_t[:,0], xyz_t[:,1], xyz_t[:,2], c= 'b', marker='x')

    # ax.scatter(xyz_1[0], xyz_1[1], xyz_1[2], c='r', marker='o')
    # ax.scatter(xyz_2[0], xyz_2[1], xyz_2[2], c='g', marker='o')
    # ax.scatter(xyz_t[0], xyz_t[1], xyz_t[2], c= 'b', marker='x')


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def multiple_plots(tracked, vicon):

    fig, axs = plt.subplots(3)

    axs[0].plot(np.arange(0,len(tracked)),tracked[:,0],color='r', label='tracked')
    axs[0].plot(np.arange(0,len(tracked)),vicon[:,0]*-1,color='b', label='vicon')
    axs[0].legend(loc="upper left")
    axs[0].set_title('X')
    axs[1].plot(np.arange(0,len(tracked)),tracked[:,1], color='r',label='tracked')
    axs[1].plot(np.arange(0,len(tracked)),vicon[:,1], color='b', label='vicon')
    axs[1].legend(loc="upper left")
    axs[1].set_title('Y')
    axs[2].plot(np.arange(0,len(tracked)),tracked[:,2],color='r',label='tracked')
    axs[2].plot(np.arange(0,len(tracked)),vicon[:,2]*-1,color='b', label='vicon')
    axs[2].legend(loc="upper left")
    axs[2].set_title('Yaw')
    plt.show()
