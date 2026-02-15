#!/usr/bin/env python
import numpy as np
from parameter_scan import param_scan


if __name__ == "__main__":
    alphas_omegas = set()

    alphas_deg = np.array([i for i in range(0,16)])
    alphas_rad = np.deg2rad(alphas_deg)
    omegas = np.arange(1,10,0.1)
    alphas_omegas.update([(a,w) for a in alphas_rad for w in omegas])

    alphas_deg = np.arange(0,15,0.1)
    alphas_rad = np.deg2rad(alphas_deg)
    omegas = [i for i in range(1,11)]
    alphas_omegas.update([(a,w) for a in alphas_rad for w in omegas])

    theta0 = np.deg2rad(0.01)
    p0=0
    gamma = 0.1


    param_scan(theta0, p0, alphas_omegas, strob=False, gamma=gamma)
