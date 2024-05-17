from casadi import *
from utils import cw_pose, cw_v_init
from numpy.linalg import norm
from tqdm import tqdm
import numpy as np

def enforce_convex_hull_from_points(normals, points, opti, X, min_station_distance):
    """enforce the convex hull obstacle given face noramls and centroids on the opti stack

    Args:
        normals (np.ndarray(num_normals, 3)): list of 3d vectors with convex hull face normals
        points (np.ndarray(num_centroids, 3)): list of 3d vectors containing triangular mesh face centroids
        opti (Opti): opti stack object for ocp
        X (MX(num_timesteps, 6)): state vector 
        min_station_distance (float): minimum distance of state vector configurations from the convex hull being enforced
    """
    # get dimensions
    num_timesteps = len(X)
    num_normals = normals.shape[0]

    # for each state timestep we apply the convex hull keepout constraint
    for j in tqdm(range(num_timesteps)):

        # # get point to evaluate constraint
        x = X[j] # state at timestep j (just position)

        # create a convex hull keepout constraint for each time step:
        dot_max = -1 # we can instantiate the max dot product as -1 because dot products less than zero do not satisfy the constraint (we take maximum)
        for i in range(num_normals):

            # first retrieve parameters for each face instance
            n = normals[i,:]/norm(normals[i,:]) # normalized face normal
            p = points[i,:] # centroid corresponding to face normal

            # only one dot product must be greater than zero so we take the maximum value
            # of all of them to use as the constraint (for each timestep)
            dot_max = fmax(dot_max, n[0]*(x[0]-p[0]) + n[1]*(x[1]-p[1]) + n[2]*(x[2]-p[2])) # Given convexity, pull out the closest face to x (state)
        
        # if max dot product value is above zero, then constraint is met (only one needs to be greater)
        try: opti.subject_to(dot_max > min_station_distance)
        except Exception as e:
            print('max, min distance, len(x), j, x, e')
            print(dot_max)
            print(min_station_distance)
            print(len(X))
            print(j)
            print(x)
            print(e)

def enforce_station_convex_hull(opti, K, IPs, T, obs, min_station_distance=1, nT=3):
    """enforce convex hull for the station obstacle given knot points, intermediate points, and minimum station proximity

    Args:
        opti (Opti): opti stack object for ocp
        K (np.array(N,3)): knot poitns
        IPs (MX): intermediate points (between knot points)
        T (MX): time intervals for each trajectory between knot points
        obs (list): list of tuples containing normals and points for each face of the station
        min_station_distance (float): minimum distance of state vector configurations from the convex hull being enforced
        nT (int, optional): number of interpolated points along each drift arc
    """
    # initialize path discretization
    nIP = IPs.shape[0]
    X = []
    X.extend([IPs[i,:] for i in range(nIP)])

    n_knots = K.shape[0]-1
    for i in range(n_knots-1):
        # get knot points via starts and ends
        starts = [K[i,:], IPs[i,:]]
        ends = [IPs[i,:], K[i+1,:]]

        # get time periods
        ts = [T[i*2], T[i*2+1]]

        # get path discretization
        for i in range(2):
            dt = ts[i]/(nT+2)
            v0 = cw_v_init(starts[i], ends[i], ts[i])
            for j in range(nT+2):
                X.append(cw_pose(starts[i], v0, dt*(j+1)))

    for oi, o in enumerate(obs):
        normals, points = o
        print(f'Enforcing {len(X)} timesteps for Obstacle {oi+1}/{len(obs)}...')
        enforce_convex_hull_from_points(normals, points, opti, X, min_station_distance)