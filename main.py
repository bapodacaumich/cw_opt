from tqdm import tqdm
from casadi import *
from cw_ocp import ocp_wrapper_noobs
from cw_ip_ocp import ocp_wrapper_intermediate
import numpy as np

def no_obs():
    view_distances = np.arange(0.5, 5.0, 0.5)
    localities = [True, False]
    max_drift_periods = 10**np.arange(1, 4, 0.5)
    for view_distance in tqdm(view_distances):
        for locality in localities:
            for max_drift_period in max_drift_periods:
                ocp_wrapper_noobs(str(view_distance)+'m', locality, filename='T', T_max=max_drift_period)

def intermediate():
    view_distances = np.arange(0.5, 5.0, 0.5)
    localities = [True, False]
    max_drift_periods = 10**np.arange(1, 4, 0.5)
    for view_distance in tqdm(view_distances):
        for locality in localities:
            for max_drift_period in max_drift_periods:
                ocp_wrapper_intermediate(
                    str(view_distance)+'m',
                    locality,
                    T_max=max_drift_period
                )

if __name__ == "__main__":
    no_obs()