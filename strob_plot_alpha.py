#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt

plots_dir = "Plots/strob_plots_varying_alpha/"

with h5py.File("Data/dissip_trajectories.h5", "r") as file:
    alphas_deg = np.arange(0,90,0.2)
    omegas = [i for i in range(1,11)]
    #omegas = [5]

    for omega in omegas:
        alpha_array = list()
        theta_array = list()
        for alpha in alphas_deg:
            grp = file[f"alpha{alpha:05.2f}/omega{omega:06.3f}/init00.0_00.0_0.1"]

            theta = grp["theta"][:]
            t = grp["t"][:]
            theta = (np.array(theta) + np.pi) % (2*np.pi) - np.pi  #  Plotting theta in the range -pi to pi

            dt_estimate = np.median(np.diff(t))
            T = 2*np.pi/omega
            nT = np.round(t / T).astype(int)
            mask = np.abs(t - nT*T) < dt_estimate/2
            theta = theta[mask][-50:]


            alpha = [alpha]*np.size(theta)
            theta_array.extend(theta)
            alpha_array.extend(alpha)

        plt.figure()
        plt.scatter(alpha_array, theta_array, s=0.1, color="black")
        plt.xlabel(r"$\alpha\,(deg)$")
        plt.ylabel("Stroboscopic sampling of " + r"$\theta(t=nT)$")
        plt.ylim(-np.pi, np.pi)
        plt.title("Stroboscopic "
                  r"$\theta$" " vs. " r"$\alpha$"
                  "\n" rf"$\omega={omega:05.2f}\, rad/s$"
                  )
        plt.grid(False)

        plt.savefig(f"{plots_dir}dissip_{omega:04.1f}.jpg")
        print(f"{plots_dir}{omega:04.1f}.jpg")
        plt.close()
