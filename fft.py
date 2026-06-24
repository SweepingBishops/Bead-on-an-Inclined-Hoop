#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt

plots_dir = "Plots/fft/"

#alphas_deg = range(57, 66, 1)
omegas = np.arange(3.0,3.4,0.02)
alphas_deg = [59]
#omegas = [2.24]

with h5py.File("Data/dissip_trajectories.h5", "r") as file:
    for alpha in alphas_deg:
        alpha_grp = file[f"alpha{alpha:05.2f}"]
        for omega_val in omegas:
            omega_grp = alpha_grp[f"omega{omega_val:06.3f}"]
            omega = omega_grp.attrs["omega"]
            trjy_grp = omega_grp["uniform30.0_00.0_0.5"]
            theta0 = trjy_grp.attrs["theta0"]
            thetadot0 = trjy_grp.attrs["thetadot0"]
            theta = (trjy_grp["theta"][:] + np.pi) % (2*np.pi) - np.pi  #  Plotting theta in the range -pi to pi
            thetadot = trjy_grp["thetadot"]
            tau = trjy_grp["tau"]
            
            theta = theta - np.mean(theta)

            dtau = tau[1] - tau[0]  # Sampling interval
            print(f"alpha: {alpha:.2f} deg, omega: {omega:.3f} rad/s")
            print("max spacing error: ", np.max(np.abs(np.diff(tau) - dtau)))
            print()


            # FFT
            fft_vals = np.fft.rfft(theta)
            freqs = np.fft.rfftfreq(len(theta), d=dtau)
            harmonics = 2*np.pi*freqs
            amps = 2*np.abs(fft_vals)/len(theta)

            plt.figure()
            plt.plot(harmonics, amps, color="black")
            plt.xlabel(r"$\omega$")
            plt.ylabel(r"FFT amplitudes")
            plt.xticks([i for i in range(0,31)])
            plt.title(
                      "FFT plot"
                      #"\n" rf"$\alpha={alpha:05.2f}^\circ,\,gamma={init_grp.attrs['gamma']}$"
                      "\n" rf"$\alpha={alpha:05.2f}^\circ\quad \omega={omega:06.3f}\,\mathrm{{rad/s}}\quad\gamma=0.5 \mathrm{{s}}^{{-1}}$"
                      "\n" rf"$\theta_0={np.rad2deg(theta0):.2f}^\circ,\, \dot\theta_0={thetadot0}$"
                      )

            file_name = f"{plots_dir}{alpha:05.2f}_{omega:06.3f}.jpg"
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.2)
            print(file_name)
            #plt.show()
            plt.close()
