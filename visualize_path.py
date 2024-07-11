import os
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_path, plot_station
from sys import argv


def visualize_traj(dist, local, t_max, soln_folder='intermediate'):

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

def visualize_intermediate_traj(dist='1.5m', local=True, t_max=36000.0, soln_folder='intermediate'):
    if local: filetxt = dist + '_local_' + str(t_max)
    else: filetxt = dist + '_' + str(t_max)
    T = np.loadtxt(os.path.join(os.getcwd(), 'solns', soln_folder, filetxt + '_t.csv'))
    X = np.loadtxt(os.path.join(os.getcwd(), 'solns', soln_folder, filetxt + '_x.csv'), delimiter=',')
    axes = plot_path(T, X, distance=dist, local=local)
    axes = plot_station(axes)
    axes.set_axis_off()
    plt.show()

def visualize_intermediate_traj_compare(dist1, local1, t_max1, dist2, local2, t_max2, soln_folder='intermediate'):
    if local1: filetxt1 = dist1 + '_local_' + str(t_max1)
    else: filetxt1 = dist2 + '_' + str(t_max2)
    if local2: filetxt2 = dist2 + '_local_' + str(t_max2)
    else: filetxt2 = dist2 + '_' + str(t_max1)
    T1 = np.loadtxt(os.path.join(os.getcwd(), 'solns', soln_folder, filetxt1 + '_t.csv'))
    X1 = np.loadtxt(os.path.join(os.getcwd(), 'solns', soln_folder, filetxt1 + '_x.csv'), delimiter=',')
    T2 = np.loadtxt(os.path.join(os.getcwd(), 'solns', soln_folder, filetxt2 + '_t.csv'))
    X2 = np.loadtxt(os.path.join(os.getcwd(), 'solns', soln_folder, filetxt2 + '_x.csv'), delimiter=',')
    axes = plot_path(T1, X1, distance=dist1, local=local1)
    axes = plot_path(T2, X2, distance=dist2, local=local2, axes=axes)
    axes = plot_station(axes)
    axes.legend()
    axes.set_axis_off()
    plt.show()

def visualize_debug_traj(debug_folder='run2', dist='1.5m', local=True):

    highest_iter = 0
    for file in os.listdir(os.path.join(os.getcwd(), 'debug', debug_folder)):
        print(file)
        print(file[2:-4])
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
        print('Example: \npython visualize_path.py -i 1.5m True 1000.0')
    elif argv[1] == '-d':
        visualize_debug_traj(debug_folder=argv[2], dist='2.0m', local=True)
    elif argv[1] == '-i':
        local_in = (argv[3]=='True' or argv[3]=='true' or argv[3] == 'T' or argv[3] == 't')
        if len(argv) == 5:
            visualize_intermediate_traj(argv[2], local_in, argv[4])
        elif len(argv) == 6:
            visualize_intermediate_traj(argv[2], local_in, t_max=argv[4], soln_folder=argv[5])
    elif argv[1] == '-c':
        local_in1 = (argv[3]=='True' or argv[3]=='true' or argv[3] == 'T' or argv[3] == 't')
        local_in2 = (argv[6]=='True' or argv[6]=='true' or argv[6] == 'T' or argv[6] == 't')
        visualize_intermediate_traj_compare(argv[2], local_in1, argv[4], argv[5], local_in2, argv[7])
    else:
        local_in = (argv[2]=='True' or argv[2]=='true' or argv[2] == 'T' or argv[2] == 't')
        visualize_traj(float(argv[1]), local_in, float(argv[3]))
