#!/usr/bin/env python
import matplotlib.pyplot as plt
import h5py
import numpy as np

plots_dir = "Plots/poincare_sections/"

#alphas_deg = [i for i in range(85,90)]
alphas_deg = [60]
omegas = np.arange(5,6,0.02)
#omegas = [5, 5.5]
samples_per_period = 64

with h5py.File(f"Data/dissip_trajectories.h5", "r") as file:
    for alpha_val_deg in alphas_deg:
        for omega in omegas:
            for init_grp in file[f"alpha{alpha_val_deg:05.2f}/omega{omega:06.3f}"].values():
                
                if "uniform30.0_00.0" not in init_grp.name:
                    continue

                # --- Load full trajectory ---
                theta = init_grp["theta"][::samples_per_period]
                thetadot = init_grp["thetadot"][::samples_per_period]

                # Wrap theta
                theta = (theta + np.pi) % (2*np.pi) - np.pi

                # --- Plot ---
                plt.figure()
                plt.scatter(theta, thetadot, s=2)

                plt.xlabel(r"$\theta\;(rad)$")
                plt.ylabel(r"$thetadot\;(rad/s)$")
                plt.xlim(-np.pi, np.pi)

                alpha = np.rad2deg(init_grp.attrs["alpha"])
                omega_val = init_grp.attrs["omega"]
                gamma_val = init_grp.attrs["gamma"]
                theta0 = np.rad2deg(init_grp.attrs["theta0"])
                thetadot0 = init_grp.attrs["thetadot0"]

                plt.title(
                    "Poincaré section\n"
                    rf"$\alpha = {alpha:.1f}^\circ,\ \omega = {omega_val:.2f}\,rad/s,\ \gamma = {gamma_val}$"
                    "\n"
                    rf"$\theta_0 = {theta0:.1f}^\circ,\ \dot\theta_0 = {thetadot0}$"
                )

                file_path = plots_dir + f"{alpha:04.1f}_{omega_val:06.3f}.png"
                plt.savefig(file_path)
                print(file_path)
                #plt.show()
                plt.close()
