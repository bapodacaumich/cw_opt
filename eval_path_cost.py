import os
import numpy as np
from math import floor
from sys import argv
from utils import compute_path_cost, dv_to_fuel

def eval_path_cost(sol_file, view_distance, local):

    for file in os.listdir(os.path.join(os.getcwd(), 'ccp_paths')):
        if str(view_distance) == file[:4]:
            if ((file[5] == 'l') and local) or ((file[5] != 'l') and not local):
                knotfile=os.path.join(os.getcwd(), 'ccp_paths', file)
                break

    print('Importing Knot File: ', knotfile)

    knot_points = np.loadtxt(knotfile, delimiter=',')[:,:3] # get positions, not orientations

    sol_path = os.path.join(os.getcwd(), 'solns', sol_file)
    T = np.loadtxt(sol_path)
    sec = floor(sum(T))
    min = floor(sec/60)
    hr = floor(min/60)
    sec -= min*60
    min -= hr*60

    print(f'Total time: {hr} hours, {min} minutes, {sec} seconds')
    dv_tot = compute_path_cost(T, knot_points, square=False)

    print("Total Delta-V: ", dv_tot)
    print('Fuel cost: ', dv_to_fuel(dv_tot)*1000)

if __name__ == "__main__":
    local_in = (argv[3]=='True' or argv[3]=='true' or argv[3] == 'T' or argv[3] == 't')
    eval_path_cost(argv[1], argv[2], local_in)