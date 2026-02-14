#!/usr/bin/env python
from time_evolution import velocity_verlet
import storage_setup
import numpy as np
import h5py

initial_conditions = [
        (np.deg2rad(0.1), 0),
        ]


omegas = np.arange(0.01, 10, 0.01)


with h5py.File("Data/poincare_trajectories.h5", "a") as file:
    for alpha_grp in file.values():
        for omega in omegas:

            omega_grp = storage_setup.get_or_create_group(alpha_grp, f"omega{omega:06.3f}", attrs={"omega":omega})
            discard_initial_time = 2*np.pi/omega * 100  # Discard the first 100 cycles
            t_fin = discard_initial_time + 301 * 2*np.pi/omega  # To get at least 300 points after the discarding in the strob result

            for i, init in enumerate(initial_conditions):
                init_grp = storage_setup.get_or_create_group(omega_grp, f"init0_short_time", attrs={"theta0": init[0], "p0": init[1], "dt": np.round(2*np.pi/(100*omega), 3)})
                omega_val = init_grp.attrs["omega"]
                alpha = init_grp.attrs["alpha"]
                t, theta, p = velocity_verlet(
                        init[0], init[1], t_fin,
                        omega_val, alpha,
                        discard_initial_time = discard_initial_time,
                        strob=True,
                        strob_time=2*np.pi/omega,
                        dt = 2*np.pi/(100*omega)  # Should have at least 100 steps 
                        )                         # per time period
                        

                storage_setup.create_or_overwrite_dataset(
                        init_grp, "theta", theta
                        )
                storage_setup.create_or_overwrite_dataset(
                        init_grp, "p", p
                        )
                
                print("-"*75)
                print(f"Finished alpha={np.rad2deg(alpha):.2f} deg, omega={omega:.2f} rad/s\tLength={np.shape(theta)}")

        print("="*100)
