from utils import get_initial_intermediate, compute_path_cost_intermediate, load_knots, dv_to_fuel
from os.path import join
from os import getcwd
import numpy as np

if __name__ == "__main__":
    T_max = 600
    knot_points = load_knots('2.0m')
    n_knots = knot_points.shape[0]
    n_intermediate = n_knots - 1
    n_drift = (n_knots + n_intermediate) - 1 # drift periods between each knot and intermediate point
    Tinit = np.ones((n_drift,1))*T_max/n_drift/2
    print(Tinit.shape)
    Tinit = np.loadtxt(join(getcwd(), 'solns', 'obs', '2.0m_local_600.0_t.csv'))
    print(Tinit)
    X = get_initial_intermediate(Tinit, knot_points, use_numpy=True)
    dv = compute_path_cost_intermediate(Tinit, knot_points, intermediate_points=X, square=False)
    cost = dv_to_fuel(dv)
    print(f'DV, Fuel = {dv} m/s, {cost} g')