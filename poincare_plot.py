#!/usr/bin/env python
import matplotlib.pyplot as plt
import h5py
import numpy as np

plots_dir = "Plots/poincare_sections/"

with h5py.File("Data/dissip_trajectories.h5", "r") as file:
    alphas_deg = [15]
    omegas = [6]
    for alpha_val_deg in alphas_deg:
        for omega in omegas:
            for init_grp in file[f"alpha{alpha_val_deg:05.2f}/omega{omega:06.3f}"].values():
                if "init00.0_00.0_0.1" not in init_grp.name:
                    continue

                theta_full = init_grp["theta"][:]
                theta_full = (np.array(theta_full) + np.pi) % (2*np.pi) - np.pi  #  Plotting theta in the range -pi to pi
                t = init_grp["t"][:]
                omega = init_grp.attrs["omega"]

                dt_estimate = np.median(np.diff(t))
                T = 2*np.pi/omega
                nT = np.round(t / T).astype(int)
                mask = np.abs(t - nT*T) < dt_estimate/2
                theta = theta_full[mask]
                p_full = init_grp["p"][:]
                p = p_full[mask]

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
                        rf"$\alpha = {alpha:.1f}^\circ,\ \omega = {omega:.2f}\,rad/s,\ $"
                        rf"$\theta_0 = {theta0:.1f}^\circ,\ p_0 = {p0}$"
                    )

                #plt.savefig(plots_dir + f"{alpha:04.1f}_{omega:05.2f}" + ".svg")

                plt.figure()
                plt.plot(t, theta_full)
                plt.ylabel(r"$\theta\,(rad)$")
                plt.xlabel(r"t")
                plt.ylim(-np.pi, np.pi)

                plt.figure()
                plt.plot(t, p_full)
                plt.ylabel(r"$p_\theta$")
                plt.xlabel(r"t")
                plt.ylim(-np.pi, np.pi)

                plt.show()
                plt.close()
