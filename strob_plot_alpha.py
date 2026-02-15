#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt

plots_dir = "Plots/strob_plots_varying_alpha/"

with h5py.File("Data/poincare_trajectories.h5", "r") as file:
    alphas_deg = np.arange(0,15,0.01)
    omegas = [i for i in range(1,11)]


    for omega in omegas:
        alpha_array = list()
        theta_array = list()
        for alpha in alphas_deg:
            grp = file[f"alpha{alpha:05.2f}/omega{omega:06.3f}/init0_short_time"]

            theta = grp["theta"][-100:]
            theta = (np.array(theta) + np.pi) % (2*np.pi) - np.pi  #  Plotting theta in the range -pi to pi
            alpha = [alpha]*np.size(theta)


            theta_array.extend(theta)
            alpha_array.extend(alpha)

        plt.figure()
        plt.scatter(alpha_array, theta_array, s=0.1, color="black")
        plt.xlabel(r"$\alpha\,(deg)$")
        plt.ylabel("Stroboscopic sampling of " + r"$\theta(t=nT)$")
        #plt.ylim(-np.pi, np.pi)
        plt.title("Stroboscopic "
                  r"$\theta$" " vs. " r"$\alpha$"
                  "\n" rf"$\omega={omega:05.2f}\, rad/s$"
                  )

        plt.savefig(f"{plots_dir}{omega:04.1f}.jpg")
        print(f"{plots_dir}{omega:04.1f}.jpg")
        plt.close()
