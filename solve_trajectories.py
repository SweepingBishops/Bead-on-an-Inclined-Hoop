#!/usr/bin/env python
import numpy as np
from parameter_scan import param_scan


if __name__ == "__main__":
    alphas_deg = np.array([1,2])
    alphas_rad = np.deg2rad(alphas_deg)
    #omegas = np.arange(0.01,10,0.01)
    omegas = [0.01] + [i for i in range(1,11)]
    #theta0 = np.deg2rad(0.01)
    theta0 = 0.5
    p0=0

    param_scan(theta0, p0, alphas_rad, omegas, strob=False)
