# clohessy wiltshire intermediate point method
from casadi import *
import numpy as np
from utils import compute_path_cost, load_station_mesh
import os
from sys import argv
from constraints import enforce_station_convex_hull
from time import perf_counter

def ocp_obs(knot_points, T_max=36000.0, debug=False):
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
    n_drift = n_knots - 1 # drift periods between each knot and intermediate point
    print('Number of drift periods: ', n_drift)
    T = opti.variable(n_drift,1)    # drift periods between each knot and intermediate point

    # constrain path to maintain keepout region
    obs = load_station_mesh()
    print('Enforcing station convex hull...')
    tstart = perf_counter()
    enforce_station_convex_hull(opti, knot_points, T, obs)
    print('Done!')
    print('Time elapsed: ', perf_counter()-tstart, 's')

    # # constrain time intervals above 0 and total below T_max
    opti.subject_to(sum1(sum2(T)) <= T_max)
    opti.subject_to(T > 0)

    # compute path cost
    dv_tot = compute_path_cost(T, knot_points, square=True)

    # minimize total delta-v
    opti.minimize(dv_tot)

    # set initial drift period values to a fraction of T_max/2
    Tinit = DM.ones(n_drift,1)*T_max/n_drift/2
    opti.set_initial(T, Tinit)

    # # debugger
    # if debug:
        # print('Debug Mode On')
        # run_num=0
        # for file in os.listdir(os.path.join(os.getcwd(), 'debug')):
            # if 'run' in file:
                # run_num +=1
        # debug_dir = os.path.join(os.getcwd(), 'debug', 'run'+str(run_num))
        # opti.callback(lambda i: debug_save_vars_intermediate(opti, T, dv_tot, debug_dir, i))

        # print('Debug run: ', run_num)

    ## solver
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-3, 'ipopt.max_iter':5000, 'ipopt.print_level': 7}
    opti.solver('ipopt', opts)
    try: sol = opti.solve()
    except RuntimeError:
        print('RUNTIME ERROR, will save non-converged values anyways')
        return opti.debug.value(T)

    return sol.value(T)

def ocp_wrapper_obs(view_distance, local, save_dir='obs', T_max=1000.0, debug=False):

    for file in os.listdir(os.path.join(os.getcwd(), 'ccp_paths')):
        if str(view_distance) == file[:4]:
            if ((file[5] == 'l') and local) or ((file[5] != 'l') and not local):
                knotfile=os.path.join(os.getcwd(), 'ccp_paths', file)
                break

    knot_points = np.loadtxt(knotfile, delimiter=',')[:,:3] # get positions, not orientations
    # knots = filter_path_na(path) # get rid of configurations with nans

    save_folder = os.path.join(os.getcwd(), 'solns', save_dir)
    
    if not os.path.exists(save_folder): os.mkdir(save_folder)

    sol_t = ocp_obs(knot_points, T_max=T_max, debug=debug)

    locality = ''
    if local:
        locality = '_local'

    np.savetxt(os.path.join(save_folder, view_distance + locality + '_' + str(T_max) + '_t.csv'), sol_t, delimiter=",")

if __name__ == "__main__":
    if argv[1] == '-h':
        print('python cw_ocp.py view_distance locality max_drift_period')
        print('DEFAULT: python cw_ip_ocp.py 1.5m True intermediate 1000.0')
    elif len(argv) == 3:
        print('Debug Mode Activated!')
        local_in = (argv[2]=='True' or argv[2]=='true' or argv[2] == 'T' or argv[2] == 't')
        ocp_wrapper_obs(argv[1], local_in, debug=True)
    else:
        local_in = (argv[2]=='True' or argv[2]=='true' or argv[2] == 'T' or argv[2] == 't')
        ocp_wrapper_obs(argv[1], local_in, save_dir=argv[3], T_max=float(argv[4]))
