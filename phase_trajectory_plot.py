#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py

dissip = "dissip_"
data_file_path = f"Data/{dissip}trajectories.h5"
phase_plots_path = "Plots/phase_plots/"
time_plots_path = "Plots/time_series_plots/"

with h5py.File(data_file_path, "r") as file:
    alphas = [i for i in range(16)]
    omegas = [i for i in range(1,11)]
    for alpha_val in alphas:
        for omega in omegas:
            for init_grp in file[f"alpha{alpha_val:05.2f}/omega{omega:06.3f}"].values():
                if  "init00.0_00.0_0.1" not in init_grp.name:
                    continue
                alpha = init_grp.attrs["alpha"]
                theta0 = init_grp.attrs["theta0"]
                p0 = init_grp.attrs["p0"]

                theta = init_grp["theta"][:]
                theta = (np.array(theta) + np.pi) % (2*np.pi) - np.pi  #  Plotting theta in the range -pi to pi
                p = init_grp["p"][:]

                file_name = f"{np.rad2deg(alpha):03.1f}_{omega:04.1f}-{np.rad2deg(theta0):04.1f}_{p0:04.1f}.jpg"
                plt.figure(figsize=(10,7))
                plt.plot(theta, p, lw=0.2, color="navy", alpha=0.4)
                plt.xlabel(r"$\theta\,(rad)$")
                plt.ylabel(r"$p_\theta\, (m^2 rad/s)$")
                plt.xlim(-np.pi, np.pi)
                plt.ylim(-np.pi, np.pi)
                plt.title("Phase Space Trajectory\n"
                          rf"$\alpha={np.rad2deg(alpha):03.1f}^\circ\, \omega={omega:03.1f}$"
                          "\n"
                          rf"$\theta_0={np.rad2deg(theta0):04.1f}^\circ\, p_0={p0:04.1f}$"
                          )

                plt.savefig(phase_plots_path + dissip + file_name, bbox_inches="tight")
                plt.close()
                
                if "dissip" not in data_file_path:
                    t = init_grp["t"][:]
                else:
                    dt = init_grp.attrs["dt"]
                    t = [n*dt for n in range(len(theta))]
                plt.figure(figsize=(10,7))
                plt.plot(t, theta, color="navy")
                plt.xlabel(r"$t\,(sec)$")
                plt.ylabel(r"$\theta\,(rad)$")
                plt.ylim(-np.pi/2, np.pi/2)
                plt.title(r"$\theta\,vs. t$"
                          "\n"
                          rf"$\alpha={np.rad2deg(alpha):03.1f}^\circ\, \omega={omega:03.1f}$"
                          "\n"
                          rf"$\theta_0={np.rad2deg(theta0):04.1f}^\circ\, p_0={p0:04.1f}$"
                          )
                plt.savefig(time_plots_path + dissip + file_name, bbox_inches="tight")
                plt.close()
                #print(dissip + file_name + " Done")
                print(init_grp.name)
