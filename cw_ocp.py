from casadi import Opti, DM
import numpy as np
from utils import compute_path_cost
import os
from sys import argv

def ocp_no_obs(knot_points, T_max=99999.0):

    n_knots = knot_points.shape[0]
    opti = Opti()

    # time intervals for each traj between knot points
    T = opti.variable(n_knots-1,1)

    # constrain time intervals above 0 and below T_max
    opti.subject_to(T <= T_max)
    opti.subject_to(T > 0)

    # compute path cost
    dv_tot = compute_path_cost(T, knot_points)

    # minimize total deltav
    opti.minimize(dv_tot)

    # warm start with reasonable values
    opti.set_initial(T, DM.ones(n_knots-1,1))

    ## solver
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-9}
    opti.solver('ipopt', opts)
    sol = opti.solve()

    return sol.value(T)

def ocp_wrapper_noobs(view_distance, local, save_dir='solns', filename='soln_t', T_max=1000.0):

    for file in os.listdir(os.path.join(os.getcwd(), 'ccp_paths')):
        if str(view_distance) == file[:4]:
            if ((file[5] == 'l') and local) or ((file[5] != 'l') and not local):
                knotfile=os.path.join(os.getcwd(), 'ccp_paths', file)
                break

    knot_points = np.loadtxt(knotfile, delimiter=',')[:,:3] # get positions, not orientations
    # knots = filter_path_na(path) # get rid of configurations with nans

    save_folder = os.path.join(os.getcwd(), save_dir)
    
    if not os.path.exists(save_folder): os.mkdir(save_folder)

    # uinum = 0
    # uitxt = ''
    # while (os.path.exists(os.path.join(save_folder, filename + uitxt + '.csv'))):
    #     if uitxt != '':
    #         uinum += 1
    #     uitxt = '_' + str(uinum)

    sol_t = ocp_no_obs(knot_points, T_max=T_max)

    locality = ''
    if local:
        locality = '_local'

    np.savetxt(os.path.join(save_folder, filename + '_' + view_distance + locality + '_' + str(T_max) + '.csv'), sol_t)

if __name__ == "__main__":
    if argv[1] == '-h':
        print('python cw_ocp.py view_distance locality max_drift_period')
        print('DEFAULT: python cw_ocp.py 1.5m True 1000.0')
    else:
        local_in = (argv[2]=='True' or argv[2]=='true' or argv[2] == 'T' or argv[2] == 't')
        ocp_wrapper_noobs(argv[1], local_in, T_max=float(argv[3]))