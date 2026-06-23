#!/usr/bin/env python
from multiprocessing import Pool
import os, signal
import numpy as np
import h5py
from tqdm import tqdm
from time_evolution import time_evolve_rk
import storage_setup

discard_tau = 100*2*np.pi
data_tau = 400*2*np.pi
samples_per_period = 64

g = 9.8
R = 0.355/2

def compute_rk(params):
    alpha, omega, theta0, thetadot0, gamma = params

    A = g/(R * omega**2)
    B = gamma/omega

    uniform = time_evolve_rk(
            theta0, thetadot0, data_tau,
            alpha, A, B,
            discard_tau = discard_tau,
            gamma=gamma,
            samples_per_period=samples_per_period,
            )

    return uniform, alpha, omega, gamma

# def compute_verlet(params):
#     alpha, omega, theta0, p0, dt, gamma = params
# 
#     T = 2*np.pi/omega
#     dt = min(T/128, 0.01)  # To get a good resolution of the dynamics
#     discard = 100*T  # Discard initial transient behaviour
#     t_fin = discard + 1_000*T
# 
#     sol = velocity_verlet(
#             theta0, p0, t_fin,
#             omega, alpha,
#             discard_initial_time = discard,
#             dt=dt
#             )
# 
#     return sol, alpha, omega, dt, T, gamma

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def param_scan(theta0, thetadot0, alphas_omegas, gamma=0):

    param_list = [(a,w, theta0, thetadot0, gamma) for a,w in alphas_omegas]

    pool = Pool(processes=max(os.cpu_count(),1) , initializer=init_worker)
    #pool = Pool(processes=max(os.cpu_count()-1,1) , initializer=init_worker)

    try:
        if gamma == 0:
            file_path = "Data/trajectories.h5"
            print("Non-dissipative case not Implemented.")
            raise NotImplementedError
        else:
            file_path = "Data/dissip_trajectories.h5"

        with h5py.File(file_path, "a") as file:
            if gamma == 0:
                worker = compute_verlet
                print("Using velocity verlet")
            else:
                worker = compute_rk
                print("Using DOP853")

            for uniform, alpha, omega, gamma in tqdm(
                    pool.imap_unordered(worker, param_list),
                    total=len(param_list),
                    desc="Computing trajectories"
                    ):

                alpha_grp = storage_setup.get_or_create_group(file, f"alpha{np.rad2deg(alpha):05.2f}", attrs={"alpha":alpha})
                omega_grp = storage_setup.get_or_create_group(alpha_grp, f"omega{omega:06.3f}", attrs={"omega":omega})

                uniform_grp =storage_setup.get_or_create_group(omega_grp, f"uniform{np.rad2deg(theta0):04.1f}_{thetadot0:04.1f}_{gamma}", attrs={ 
                    "theta0": theta0,
                    "thetadot0": thetadot0,
                    "gamma": gamma,
                    "samples_per_period": samples_per_period,
                    })

                storage_setup.create_or_overwrite_dataset(uniform_grp, "tau", uniform[0])
                storage_setup.create_or_overwrite_dataset(
                        uniform_grp, "theta", uniform[1][0]
                        )
                storage_setup.create_or_overwrite_dataset(
                        uniform_grp, "thetadot", uniform[1][1]
                        )

            pool.close()
            pool.join()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating workers...")
        pool.terminate()
        pool.join()
        print("Workers terminated cleanly.")
        exit()

if __name__ == "__main__":
    pass
    # theta0 = np.deg2rad(0.1)
    # p0 = 0

    # #alphas_deg = np.arange(0,15,0.01)
    # alphas_deg = np.array([0])
    # alphas_rad = np.deg2rad(alphas_deg)
    # #omegas = [i for i in range(1,11)]
    # omegas = np.arange(0.01,10, 0.01)
    # alphas_omegas = [(a,w) for a in alphas_rad for w in omegas]
    # param_scan(theta0, p0, alphas_omegas, strob=True)
