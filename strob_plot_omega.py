#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt

plots_dir = "Plots/strob_plots_varying_omega/"

with h5py.File("Data/dissip_trajectories.h5", "r") as file:
    for alpha in range(0,90, 5):
        alpha_grp = file[f"alpha{alpha:05.2f}"]
        omega_array = list()
        theta_array = list()
        for omega_val in np.arange(1,11,0.1): 
            omega_grp = alpha_grp[f"omega{omega_val:06.3f}"]
            omega = omega_grp.attrs["omega"]
            init_grp = omega_grp["init00.0_00.0_0.1"]
            # for init_grp in omega_grp.values():
            theta = init_grp["theta"][:]
            t = init_grp["t"][:]
            theta = (np.array(theta) + np.pi) % (2*np.pi) - np.pi  #  Plotting theta in the range -pi to pi
            dt_estimate = np.median(np.diff(t))
            
            T = 2*np.pi/omega
            nT = np.round(t / T).astype(int)
            mask = np.abs(t - nT*T) < dt_estimate/2
            theta = theta[mask][:]

            omega = [omega]*np.size(theta)
            theta_array.extend(theta)
            omega_array.extend(omega)

        plt.figure()
        plt.scatter(omega_array, theta_array, s=0.1, color="black")
        plt.xlabel(r"$\omega\,(rad/s)$")
        plt.ylabel("Stroboscopic sampling of " + r"$\theta(t=nT)$")
        plt.title("Stroboscopic "
                  r"$\theta$" " vs. " r"$\omega$"
                  "\n" rf"$\alpha={alpha:05.2f}^\circ,\,gamma={init_grp.attrs['gamma']}$"
                  )

        file_name = f"{plots_dir}dissip_{alpha:04.1f}.jpg"
        plt.savefig(file_name)
        print(file_name)
        plt.close()
