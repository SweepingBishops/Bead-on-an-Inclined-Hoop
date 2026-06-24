#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt

plots_dir = "Plots/amplitude/"

#alphas_deg = range(0, 20, 1)
alphas_deg = [22]

with h5py.File("Data/dissip_trajectories.h5", "r") as file:
    for alpha in alphas_deg:
        alpha_grp = file[f"alpha{alpha:05.2f}"]
        omegas = list()
        amplitudes = list()
        for omega_val in np.arange(1,6,0.02): 
            omega_grp = alpha_grp[f"omega{omega_val:06.3f}"]
            omega = omega_grp.attrs["omega"]
            trjy_grp = omega_grp["uniform30.0_00.0_0.5"]
            theta0 = trjy_grp.attrs["theta0"]
            thetadot0 = trjy_grp.attrs["thetadot0"]
            theta = (trjy_grp["theta"][:] + np.pi) % (2*np.pi) - np.pi  #  Plotting theta in the range -pi to pi
            amplitude = np.max(theta) - np.min(theta)
            omegas.append(omega)
            amplitudes.append(amplitude)


        plt.figure()
        plt.plot(omegas, amplitudes, color="black")
        plt.xlabel(r"$\omega\,(rad/s)$")
        plt.ylabel(r"$A=\theta_\max-\theta_\min$")
        plt.ylim(-0.25, 2*np.pi+0.25)
        plt.title(
                  "Amplitude"" vs. " r"$\omega$"
                  #"\n" rf"$\alpha={alpha:05.2f}^\circ,\,gamma={init_grp.attrs['gamma']}$"
                  "\n" rf"$\alpha={alpha:05.2f}^\circ\quad \gamma=0.5$"
                  "\n" rf"$\theta_0={np.rad2deg(theta0):.2f}^\circ,\, \dot\theta_0={thetadot0}$"
                  )

        file_name = f"{plots_dir}{alpha:05.2f}.jpg"
        #plt.savefig(file_name, bbox_inches="tight", pad_inches=0.2)
        print(file_name)
        plt.show()
        plt.close()
