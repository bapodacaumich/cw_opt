import os
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_path
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
    plot_path(T, distance=str(dist) + 'm', local=local)

if __name__ == '__main__':
    if argv[1] == '-h':
        print('Example: \npython visualize_path.py 1.5 True 1000.0')
    local_in = (argv[2]=='True' or argv[2]=='true' or argv[2] == 'T' or argv[2] == 't')
    visualize_traj(float(argv[1]), local_in, float(argv[3]))