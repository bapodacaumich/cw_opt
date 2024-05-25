from casadi import sin, cos, sqrt
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import linalg as la
from stl import mesh
from mpl_toolkits import mplot3d

def cw_v_init(start, end, t, n=1.1288e-3):
    """compute initial velocity vector for drift trajectory given two points (start, end), period between two points (t) and orbital rate (n)

    Args:
        start (np.array(3,)): start pose
        end (np.array(3,)): end pose
        t (float): path length (time) between start and end
        n (float, optional): orbiting body angular rate. Defaults to 1.1288e-3.
    """
    dx = end[0]
    dy = end[1]
    dz = end[2]
    x = start[0]
    y = start[1]
    z = start[2]

    sigma3 = (cos(n*t))**2
    sigma2 = (sin(n*t))**2
    sigma1 = 4*sigma3 - 8*cos(n*t) + 4*sigma2 - 3*n*t*sin(n*t) + 4
    vx = -n*(2*dy - 2*y - 2*dy*cos(n*t) - 4*dx*sin(n*t) + 2*y*cos(n*t) + 4*x*sin(n*t) + 3*dx*n*t - 3*n*t*x*cos(n*t))/sigma1
    vy = -n*(8*x - 2*dx + 2*dx*cos(n*t) - dy*sin(n*t) - 14*x*cos(n*t) + y*sin(n*t) + 6*x*sigma3 + 6*x*sigma2 - 6*n*t*x*sin(n*t))/sigma1
    vz = n*(dz - z*cos(n*t))/sin(n*t)

    return vx, vy, vz

def cw_pose(x0, v0, t, n=1.1288e-3):
    x = x0[0]
    y = x0[1]
    z = x0[2]
    vx = v0[0]
    vy = v0[1]
    vz = v0[2]
    sigma1 = cos(n*t)-1
    xe = vx*sin(n*t)/n - x*(3*cos(n*t) - 4) - 2*vy*sigma1/n
    ye = y + x*(6*sin(n*t) - 6*n*t) + vy*(4*sin(n*t) - 3*n*t)/n + 2*vx*sigma1/n
    ze = z*cos(n*t) + vz*sin(n*t)/n
    return xe, ye, ze

def dv_to_fuel(dv, m=5, isp=80):
    """_summary_

    Args:
        dv (float): delta-v in m/s
        m (float, optional): dry mass of spacecraft (kg). Defaults to 5.
        isp (float, optional): specific impulse of engine. Defaults to 80.
    Returns:
        cost (float): fuel cost to make dv (kg)
    """
    g0 = 9.81 # standard gravity
    m0 = m * np.exp(dv/(isp*g0))
    cost = m0 - m
    return cost

def cw_v_end(start, v_init, t, n=1.1288e-3):
    """get end velocity of drift trajectory given start, initial velocity and time

    Args:
        start (np.array(3)): start point
        v_init (MX.array(3)): start velocity
        t (float or casadi symbolic): drift time
        n (float, optional): orbital rate of target object. Defaults to 1.1288e-3.

    Returns:
        tuple: end velocity vector components (x,y,z)
    """
    # numpy implementation
    x = start[0]
    y = start[1]
    z = start[2]
    vx = v_init[0]
    vy = v_init[1]
    vz = v_init[2]
    vx_end = vx*cos(n*t) + 2*vy*sin(n*t) + 3*n*x*sin(n*t)
    vy_end = vy*(4*cos(n*t)-3) - 2*vx*sin(n*t) + 6*n*x*(cos(n*t)-1)
    vz_end = vz*cos(n*t) - n*z*sin(n*t)

    return vx_end, vy_end, vz_end

def filter_path_na(path):
    """
    remove waypoints with nan from path
    """
    knot_bool = ~np.isnan(path)[:,-1]
    return path[knot_bool,:]

def compute_path_cost(T, knot_points, square=True):
    """sum delta-v between each traj including initial

    TODO: Vectorize

    Args:
        T (opti): opti variable
        knot_points (np.array): list of knot points
    """

    n_knots = knot_points.shape[0]
    last_v = [0,0,0]
    dv_tot = 0
    for i in range(n_knots-1):
        last_knot = knot_points[i]
        next_knot = knot_points[i+1]
        cur_T = T[i]
        vx, vy, vz = cw_v_init(last_knot, next_knot, cur_T)
        if square: dv_tot += (last_v[0]-vx)**2 + (vy-last_v[1])**2 + (vz-last_v[2])**2
        else: dv_tot += sqrt((last_v[0]-vx)**2 + (vy-last_v[1])**2 + (vz-last_v[2])**2)
        vx_end, vy_end, vz_end = cw_v_end(last_knot, [vx, vy, vz], cur_T)
        last_v = [vx_end, vy_end, vz_end]

    return dv_tot

def compute_path_cost_intermediate(T, knot_points, intermediate_points, square=True):

    # ensure sizes are correct
    # print(knot_points.shape[0], intermediate_points.shape[0], T.shape[0])
    # assert (knot_points.shape[0]-1)*2 == intermediate_points.shape[0]*2 == T.shape[0]

    n_knots = knot_points.shape[0]
    last_v = [0,0,0]
    dv_tot = 0

    for i in range(n_knots-1):
        ## drift from knot point to intermediate point
        last_knot = knot_points[i]
        next_knot = intermediate_points[i,:]
        cur_T = T[i*2]

        # get initial velocity for drift trajectory based on cur_T
        vx, vy, vz = cw_v_init(last_knot, next_knot, cur_T)

        # compute delta-v and add to total
        if square: dv_tot += (last_v[0]-vx)**2 + (vy-last_v[1])**2 + (vz-last_v[2])**2
        else: dv_tot += sqrt((last_v[0]-vx)**2 + (vy-last_v[1])**2 + (vz-last_v[2])**2)

        # get final velocity for drift trajectory based on cur_T
        last_v = cw_v_end(last_knot, [vx, vy, vz], cur_T)
        # vx_end, vy_end, vz_end = cw_v_end(last_knot, [vx, vy, vz], cur_T)
        # last_v = [vx_end, vy_end, vz_end]

        ## drift from intermediate point to knot point
        last_knot = intermediate_points[i,:]
        next_knot = knot_points[i+1]
        cur_T = T[i*2+1]

        # get initial velocity for drift trajectory based on cur_T
        vx, vy, vz = cw_v_init(last_knot, next_knot, cur_T)

        # compute delta-v and add to total
        if square: dv_tot += (last_v[0]-vx)**2 + (vy-last_v[1])**2 + (vz-last_v[2])**2
        else: dv_tot += sqrt((last_v[0]-vx)**2 + (vy-last_v[1])**2 + (vz-last_v[2])**2)

        # get final velocity for drift trajectory based on cur_T
        last_v = cw_v_end(last_knot, [vx, vy, vz], cur_T)
        # vx_end, vy_end, vz_end = cw_v_end(last_knot, [vx, vy, vz], cur_T)
        # last_v = [vx_end, vy_end, vz_end]
    return dv_tot

def debug_save_vars_intermediate(opti, T, dv_tot, X, debug_dir, i):
    """save all variables for debugging

    Args:
        opti (opti): opti stack object to save from
        T (opti variable): drift period symbolic variable (Vector)
        dv_tot (opti variable): total delta-v symbolic variable
        X (opti variable): intermediate points symbolic variable (Matrix)
        debug_dir (os.path): directory to save intermediate solution
        i (int): iteration number
    """
    if not os.path.exists(debug_dir): os.makedirs(debug_dir)
    np.savetxt(os.path.join(debug_dir, 'T_' + str(i) + '.csv'), opti.debug.value(T), delimiter=',')
    np.savetxt(os.path.join(debug_dir, 'X_' + str(i) + '.csv'), opti.debug.value(X), delimiter=',')
    with open(os.path.join(debug_dir, 'cost.txt'), 'a') as f:
        f.write(str(opti.debug.value(dv_tot))+'\n')

def plot_station(axes):
    translation = np.loadtxt('translate_station.txt', delimiter=',').reshape(1,1,3)
    scale = np.array([0])
    for i in range(15):
        meshfile = os.path.join(os.getcwd(), 'model', 'convex_detailed_station', str(i) + '.stl')

        # Load the STL files and add the vectors to the plot
        your_mesh = mesh.Mesh.from_file(meshfile)
        vectors = your_mesh.vectors + translation
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(vectors))
        wf = vectors.reshape(-1, 3)
        axes.plot(wf[:,0], wf[:,1], wf[:,2], 'k')

        # Auto scale to the mesh size
        scale = np.concatenate((scale, your_mesh.points.flatten()))
    axes.auto_scale_xyz(scale, scale, scale)
    return axes

def load_knots(distance, local=False):
    """load knotpoint file from distance and locality

    Args:
        distance (str): viewpoint generation distance ie '1.5m'
        local (bool, optional): if path is local. Defaults to False.

    Returns:
        (np.array): numpy array of knotpoints (N,6)
    """
    for file in os.listdir(os.path.join(os.getcwd(), 'ccp_paths')):
        if (distance == file[:4]):
            if not ((file[5] == 'l') ^ local):
                knotfile=os.path.join(os.getcwd(), 'ccp_paths', file)
    # knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv')
    # load knot points
    return np.loadtxt(knotfile, delimiter=',') # (N, 6)

def plot_path(T, X=None, n_drift=20, distance='1.5m', local=False):
    """plot path from list of drift periods

    Args:
        T (np.array): array of drift periods
        n_drift (int, optional): number of points to interpolate drift traj. Defaults to 20.
        distance (str, optional): viewpoint generation distance. Defaults to '1.5m'.
        local (bool, optional): if path uses local tsp formulation. Defaults to False.
    """
    # load knot points
    if X is None: knots = load_knots(distance, local)[:,:3]
    else:
        knotpoints = load_knots(distance, local)[:,:3]
        knots = []
        for i in range(X.shape[0]):
            knots.append(knotpoints[i,:])
            knots.append(X[i,:])
        knots.append(knotpoints[-1,:])
        knots = np.array(knots)

    # Create a new plot
    figure = plt.figure()
    axes = figure.add_subplot(projection='3d')
    
    # plot knot points
    # axes.plot(knots[:,0], knots[:,1], knots[:,2],'k--')
    # axes.scatter(knots[:,0], knots[:,1], knots[:,2], 'rx')

    # plot subtrajectories
    n_subtraj = knots.shape[0]-1
    full_path = np.zeros((n_drift*n_subtraj, 3))
    for i in range(n_subtraj):
        last_knot = knots[i]
        next_knot = knots[i+1]
        cur_T = T[i]
        v0 = cw_v_init(last_knot, next_knot, cur_T)
        drift = np.linspace(0, cur_T, n_drift)
        for sub_i, t in enumerate(drift):
            x, y, z = cw_pose(last_knot, v0, t)
            full_path[i*n_drift + sub_i,:] = np.array([x, y, z])

    axes.scatter(knotpoints[:,0], knotpoints[:,1], knotpoints[:,2], c='tab:orange', marker='o', lw=5, label='Knot Points')
    axes.scatter(X[:,0], X[:,1], X[:,2], c='tab:purple', marker='x', lw=5, label='Intermediate Points')
    axes.plot(full_path[:,0], full_path[:,1], full_path[:,2], 'k')
    axes.plot([full_path[0,0], full_path[-1,0]], [full_path[0,1], full_path[-1,1]], [full_path[0,2], full_path[-1,2]], 'rx')
    xmin = np.min(full_path[:,0])
    xmax = np.max(full_path[:,0])
    ymin = np.min(full_path[:,1])
    ymax = np.max(full_path[:,1])
    zmin = np.min(full_path[:,2])
    zmax = np.max(full_path[:,2])
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])
    axes.set_zlim([zmin, zmax])
    axes.set_box_aspect([xmax-xmin, ymax-ymin, zmax-zmin])
    return axes

def load_mesh(mesh_file, show=False):
    """
    import mesh .stl file
    returns mesh face normals and single point on face to establish plane position
    """

    if show: print('Importing mesh: ', mesh_file)
    mesh_file = os.path.join(os.getcwd(), mesh_file)
    str_mesh = mesh.Mesh.from_file(mesh_file)

    return str_mesh.normals/la.norm(str_mesh.normals), str_mesh.v0

def load_station_mesh():
    # get station translation for ccp path
    station_offset = np.loadtxt('translate_station.txt', delimiter=',')

    # import normals and surface points
    obs = []
    for i in range(15):
        meshfile = os.path.join('model', 'convex_detailed_station', str(i) + '.stl')
        normals, points = load_mesh(meshfile)
        points += station_offset
        obs.append((normals, points))

    return obs

if __name__ == "__main__":
    # checking cw_v_init:
    v0 = cw_v_init([0,0,0], [1,1,1], 1)
    print(v0)