from casadi import sin, cos, sqrt
import matplotlib.pyplot as plt
import numpy as np
import os
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

def plot_path(T, n_drift=20, distance='1.5m', local=False):
    """plot path from list of drift periods

    Args:
        T (np.array): array of drift periods
        n_drift (int, optional): number of points to interpolate drift traj. Defaults to 20.
        distance (str, optional): viewpoint generation distance. Defaults to '1.5m'.
        local (bool, optional): if path uses local tsp formulation. Defaults to False.
    """
    # load knot points
    knots = load_knots(distance, local)[:,:3]

    # Create a new plot
    figure = plt.figure()
    axes = figure.add_subplot(projection='3d')
    
    # plot knot points
    # axes.plot(knots[:,0], knots[:,1], knots[:,2],'k--')
    axes.scatter(knots[:,0], knots[:,1], knots[:,2], 'rx')

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

    axes.plot(full_path[:,0], full_path[:,1], full_path[:,2], 'k')
    return axes

if __name__ == "__main__":
    # checking cw_v_init:
    v0 = cw_v_init([0,0,0], [1,1,1], 1)
    print(v0)