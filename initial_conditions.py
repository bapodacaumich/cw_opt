from casadi import vertcat
from os.path import join
from numpy import loadtxt
from utils import load_mesh

def convex_hull_station():
    ## Constraints and Parameters
    thrust_limit = 100.0
    fuel_cost_weight = 1.0
    g0 = 9.81
    Isp = 80

    # problem constraints
    n_states = 6
    n_inputs = 3
 
    # first and final states
    x0 = vertcat(-2.0, -3.0, 0.0, 0.0, 0.0, 0.0)
    xf = vertcat(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    # get station translation for ccp path
    station_offset = loadtxt('translate_station.txt', delimiter=',')

    # import normals and surface points
    obs = []
    for i in range(15):
        meshfile = join('model', 'convex_detailed_station', str(i) + '.stl')
        normals, points = load_mesh(meshfile)
        points += station_offset
        obs.append((normals, points))

    return obs, n_states, n_inputs, g0, Isp