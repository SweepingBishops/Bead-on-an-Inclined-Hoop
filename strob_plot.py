#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt

plots_dir = "Plots/strob_plots_varying_omega/"

with h5py.File("Data/poincare_trajectories.h5", "r") as file:
    for alpha_grp in file.values():
        alpha = np.rad2deg(alpha_grp.attrs["alpha"])
        omega_array = list()
        theta_array = list()
        for omega_grp in alpha_grp.values():
            omega = omega_grp.attrs["omega"]
            for init_grp in omega_grp.values():
                theta = init_grp["theta"][-100:]
                theta = (np.array(theta) + np.pi) % (2*np.pi) - np.pi  #  Plotting theta in the range -pi to pi
                omega = [omega]*np.size(theta)


                theta_array.extend(theta)
                omega_array.extend(omega)

        plt.figure()
        plt.scatter(omega_array, theta_array, s=0.1, color="black")
        plt.xlabel(r"$\omega\,(rad/s)$")
        plt.ylabel("Stroboscopic sampling of " + r"$\theta(t=nT)$")
        plt.title("Stroboscopic "
                  r"$\theta$" " vs. " r"$\omega$"
                  "\n" rf"$\alpha={alpha:05.2f}^\circ$"
                  )

        plt.savefig(f"{plots_dir}{alpha:04.1f}.jpg")
        print(f"{plots_dir}{alpha:04.1f}.jpg")
        plt.close()
