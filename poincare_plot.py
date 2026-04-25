#!/usr/bin/env python
import matplotlib.pyplot as plt
import h5py
import numpy as np

plots_dir = "Plots/poincare_sections/dissipative/"

alphas_deg = [30]
#omegas = np.arange(5,10.01,0.02)
omegas = [8]

with h5py.File(f"Data/dissip_trajectories.h5", "r") as file:
    for alpha_val_deg in alphas_deg:
        for omega in omegas:
            for init_grp in file[f"alpha{alpha_val_deg:05.2f}/omega{omega:06.3f}"].values():
                
                if "init30.0_00.0" not in init_grp.name:
                    continue

                # --- Load full trajectory ---
                theta = init_grp["theta"][:]
                p = init_grp["p"][:]
                t = init_grp["t"][:]

                # Wrap theta
                theta = (theta + np.pi) % (2*np.pi) - np.pi

                # --- Compute stroboscopic mask ---
                T = 2*np.pi / omega
                dt_estimate = np.median(np.diff(t))

                nT = np.round(t / T).astype(int)
                mask = np.abs(t - nT*T) < dt_estimate/2

                theta_p = theta[mask]
                p_p = p[mask]

                # --- Plot ---
                plt.figure()
                plt.scatter(theta_p, p_p, s=0.5)

                plt.xlabel(r"$\theta\;(rad)$")
                plt.ylabel(r"$p_\theta\;(m^2 rad/s)$")
                #plt.xlim(-np.pi, np.pi)

                alpha = np.rad2deg(init_grp.attrs["alpha"])
                omega_val = init_grp.attrs["omega"]
                gamma_val = init_grp.attrs["gamma"]
                theta0 = np.rad2deg(init_grp.attrs["theta0"])
                p0 = init_grp.attrs["p0"]

                plt.title(
                    "Poincaré section\n"
                    rf"$\alpha = {alpha:.1f}^\circ,\ \omega = {omega_val:.2f}\,rad/s,\ \gamma = {gamma_val}$"
                    "\n"
                    rf"$\theta_0 = {theta0:.1f}^\circ,\ p_0 = {p0}$"
                )

                plt.savefig(plots_dir + f"{alpha:04.1f}_{omega_val:05.2f}.pdf")
                plt.show()
                plt.close()
