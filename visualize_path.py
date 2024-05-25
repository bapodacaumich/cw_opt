import os
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_path, plot_station
from sys import argv


def visualize_traj(dist, local, t_max, soln_folder='solns'):

    locality=''
    if local: locality='_local'

    soln_dir = os.path.join(os.getcwd(), soln_folder)
    query_string = 'T_' + str(dist) + 'm' + locality + '_' + str(t_max)
    q_len = len(query_string)
    for file in os.listdir(soln_dir):
        if file[:q_len] == query_string:
            sol_path = os.path.join(soln_dir, file)
            break
    print('Found File: ', sol_path, '\nplotting...')
    T = np.loadtxt(sol_path)
    axes = plot_path(T, distance=str(dist) + 'm', local=local)
    axes = plot_station(axes)
    axes.legend()
    plt.show()

def visualize_debug_traj(debug_folder='run2', dist='1.5m', local=True):

    highest_iter = 0
    for file in os.listdir(os.path.join(os.getcwd(), 'debug', debug_folder)):
        if file[0] == 'T' and int(file[2:-4]) > highest_iter:
            highest_iter = int(file[2:-4])
            file_end = file[1:]

    print('Highest Iteration:', highest_iter)

    T = np.loadtxt(os.path.join(os.getcwd(), 'debug', debug_folder, 'T' + file_end), delimiter=',')
    X = np.loadtxt(os.path.join(os.getcwd(), 'debug', debug_folder, 'X' + file_end), delimiter=',')
    axes = plot_path(T, X, distance=str(dist) + 'm', local=local)
    axes = plot_station(axes)
    plt.show()

if __name__ == '__main__':
    if argv[1] == '-h':
        print('Example: \npython visualize_path.py 1.5 True 1000.0')
    elif argv[1] == '-d':
        visualize_debug_traj(debug_folder=argv[2], dist=1.5, local=True)
    else:
        local_in = (argv[2]=='True' or argv[2]=='true' or argv[2] == 'T' or argv[2] == 't')
        visualize_traj(float(argv[1]), local_in, float(argv[3]))