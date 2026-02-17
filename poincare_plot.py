#!/usr/bin/env python
import matplotlib.pyplot as plt
import h5py
import numpy as np

plots_dir = "Plots/poincare_sections/hamiltonian/"

with h5py.File(f"Data/poincare_trajectories.h5", "r") as file:
    alphas_deg = [2]
    omegas = np.arange(3, 6.1, 0.2)
    for alpha_val_deg in alphas_deg:
        for omega in omegas:
            for init_grp in file[f"alpha{alpha_val_deg:05.2f}/omega{omega:06.3f}"].values():
                if "init00.0_00.0" not in init_grp.name:
                    continue

                theta = init_grp["theta"][:]
                theta = (np.array(theta) + np.pi) % (2*np.pi) - np.pi  #  Plotting theta in the range -pi to pi
                p = init_grp["p"][:]

                plt.figure()
                plt.scatter(
                        theta,
                        p,
                        s=0.5,
                        )
                plt.xlabel(r"$\theta\;(rad)$")
                plt.ylabel(r"$p_\theta\;(m^2 rad/s)$")
                plt.xlim(-np.pi,np.pi)
                alpha = np.rad2deg(init_grp.attrs["alpha"])
                omega = init_grp.attrs["omega"]
                theta0 = np.rad2deg(init_grp.attrs["theta0"])
                p0 = init_grp.attrs["p0"]
                plt.title(
                        f"Poincaré section\n"
                        rf"$\alpha = {alpha:.1f}^\circ,\ \omega = {omega:.2f}\,rad/s$"
                        "\n"
                        rf"$\theta_0 = {theta0:.1f}^\circ,\ p_0 = {p0}$"
                    )

                plt.savefig(plots_dir + f"{alpha:04.1f}_{omega:05.2f}" + ".pdf")
                plt.show()
                plt.close()
