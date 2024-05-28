# clohessy wiltshire intermediate point method
# from casadi import Opti, DM, sum2
from casadi import *
import numpy as np
from utils import compute_path_cost_intermediate, load_station_mesh, debug_save_vars_intermediate, get_initial_intermediate
import os
from sys import argv
from constraints import enforce_station_convex_hull
from time import perf_counter

def ocp_intermediate(knot_points, T_max=36000.0, debug=False):
    """set up and solve optimal control problem for drift trajectories with obstacles using intermediate points

    Args:
        knot_points (np.array(N,3)): matrix of knot points for trajectory
        T_max (float, optional): maximum path time. Defaults to 36000.0.

    Returns:
        _type_: _description_
    """

    n_knots = knot_points.shape[0]
    opti = Opti()

    # time intervals for each traj between knot points
    T = opti.variable(n_knots*2-2,1)    # two drift periods between each knot point
    X = opti.variable(n_knots-1,3)      # one less intermediate point than knot points (between each pair of knot points)

    # constrain path to maintain keepout region
    obs = load_station_mesh()
    print('Enforcing station convex hull...')
    tstart = perf_counter()
    # enforce_station_convex_hull(opti, knot_points, X, T, obs)
    print('Done!')
    print('Time elapsed: ', perf_counter()-tstart, 's')

    # constrain time intervals above 0 and total below T_max
    opti.subject_to(sum2(T) <= T_max) # TODO: check columnwise sum is correct (T should be Nx1)
    opti.subject_to(T > 0)

    # compute path cost
    dv_tot = compute_path_cost_intermediate(T, knot_points, intermediate_points=X, square=True)

    # minimize total delta-v
    opti.minimize(dv_tot)

    # warm start with reasonable values -- middle of drift trajectory
    # Tinit = DM.ones(n_knots*2-2,1)*1
    # opti.set_initial(T, Tinit)
    # IPinit = get_initial_intermediate(Tinit, knot_points)
    # opti.set_initial(X, IPinit)


    # set initial drift period values to 1s
    opti.set_initial(T, DM.ones((n_knots-1)*2,1))

    # set initial intermediate points to next knot point
    opti.set_initial(X, knot_points[1:,:])

    # debug print solution
    # print('test:')
    # print(compute_path_cost_intermediate(DM.ones((n_knots-1)*2,1), knot_points, intermediate_points=knot_points[1:,:]))

    # debugger
    if debug:
        print('Debug Mode On')
        run_num=0
        for file in os.listdir(os.path.join(os.getcwd(), 'debug')):
            if 'run' in file:
                run_num +=1
        debug_dir = os.path.join(os.getcwd(), 'debug', 'run'+str(run_num))
        opti.callback(lambda i: debug_save_vars_intermediate(opti, T, dv_tot, X, debug_dir, i))

        print('Debug run: ', run_num)

    ## solver
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-9, 'ipopt.max_iter':5000}
    opti.solver('ipopt', opts)
    sol = opti.solve()

    return sol.value(T), sol.value(X)

def ocp_wrapper_intermediate(view_distance, local, save_dir='intermediate', T_max=36000.0, debug=False):

    for file in os.listdir(os.path.join(os.getcwd(), 'ccp_paths')):
        if str(view_distance) == file[:4]:
            if ((file[5] == 'l') and local) or ((file[5] != 'l') and not local):
                knotfile=os.path.join(os.getcwd(), 'ccp_paths', file)
                break

    knot_points = np.loadtxt(knotfile, delimiter=',')[:,:3] # get positions, not orientations
    # knots = filter_path_na(path) # get rid of configurations with nans

    save_folder = os.path.join(os.getcwd(), 'solns', save_dir)
    
    if not os.path.exists(save_folder): os.mkdir(save_folder)

    sol_t, sol_x = ocp_intermediate(knot_points, T_max=T_max, debug=debug)

    locality = ''
    if local:
        locality = '_local'

    np.savetxt(os.path.join(save_folder, view_distance + locality + '_' + str(T_max) + '_x.csv'), sol_x, delimiter=",")
    np.savetxt(os.path.join(save_folder, view_distance + locality + '_' + str(T_max) + '_t.csv'), sol_t, delimiter=",")

if __name__ == "__main__":
    if argv[1] == '-h':
        print('python cw_ocp.py view_distance locality max_drift_period')
        print('DEFAULT: python cw_ip_ocp.py 1.5m True intermediate 1000.0')
    elif len(argv) == 3:
        local_in = (argv[2]=='True' or argv[2]=='true' or argv[2] == 'T' or argv[2] == 't')
        ocp_wrapper_intermediate(argv[1], local_in, debug=True)
    else:
        local_in = (argv[2]=='True' or argv[2]=='true' or argv[2] == 'T' or argv[2] == 't')
        ocp_wrapper_intermediate(argv[1], local_in, save_dir=argv[3], T_max=float(argv[4]))