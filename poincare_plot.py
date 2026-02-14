#!/usr/bin/env python
import matplotlib.pyplot as plt
import h5py
import numpy as np

plots_dir = "Plots/poincare_sections/"

with h5py.File("Data/poincare_trajectories.h5", "r") as file:
    for alpha_grp in file.values():
        for omega_grp in alpha_grp.values():
            for init_grp in omega_grp.values():
                plt.figure()
                plt.scatter(
                        init_grp["theta"][:],
                        init_grp["p"][:],
                        s=0.5,
                        )
                plt.xlabel(r"$\theta\;(rad)$")
                plt.ylabel(r"$p_\theta\;(m^2 rad/s)$")
                alpha = np.rad2deg(init_grp.attrs["alpha"])
                omega = init_grp.attrs["omega"]
                theta0 = np.rad2deg(init_grp.attrs["theta0"])
                p0 = init_grp.attrs["p0"]
                plt.title(
                        f"Poincaré section\n"
                        rf"$\alpha = {alpha:.1f}^\circ,\ \omega = {omega:.2f}\,rad/s,\ $"
                        rf"$\theta_0 = {theta0:.1f}^\circ,\ p_0 = {p0}$"
                    )

                plt.savefig(plots_dir + f"{alpha:04.1f}_{omega:05.2f}" + ".svg")
                plt.close()
