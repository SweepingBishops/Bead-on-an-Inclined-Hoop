#!/usr/bin/env python
from multiprocessing import Pool
import os, signal
import numpy as np
import h5py
from tqdm import tqdm
from time_evolution import velocity_verlet, time_evolve_rk
import storage_setup

def compute_poincare_rk(params):
    alpha, omega, theta0, p0, strob, gamma = params

    T = 2*np.pi/omega
    discard = min(500*T, 1_000)  # Discard initial transient behaviour
    t_fin = min(discard + 500*T, 5_000)
    # Note that this won't get good results for very small omega

    t, theta, p = time_evolve_rk(
            theta0, p0, t_fin,
            omega, alpha,
            discard_initial_time = discard,
            gamma=gamma,
            strob=strob,
            strob_time=T,
            )

    return alpha, omega, theta, p, None, t, T, gamma

def compute_poincare_verlet(params):
    alpha, omega, theta0, p0, strob, gamma = params

    T = 2*np.pi/omega
    dt = min(T/100, 0.01)  # To get a good resolution of the dynamics
    discard = min(100*T, 1_000)  # Discard initial transient behaviour
    t_fin = min(discard + 300*T, 5_000)
    # Note that this won't get good results for very small omega

    t, theta, p = velocity_verlet(
            theta0, p0, t_fin,
            omega, alpha,
            discard_initial_time = discard,
            strob=strob,
            strob_time=T,
            dt=dt
            )

    return alpha, omega, theta, p, dt, None, T, gamma

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def param_scan(theta0, p0, alphas_omegas, strob=True, gamma=0):

    param_list = [(a,w, theta0, p0, strob, gamma) for a,w in alphas_omegas]

    pool = Pool(processes=max(os.cpu_count()-1,1) , initializer=init_worker)

    try:
        if strob and gamma == 0:
            file_path = "Data/poincare_trajectories.h5"
        elif strob:
            file_path = "Data/dissip_poincare_trajectories.h5"
        elif gamma == 0:
            file_path = "Data/trajectories.h5"
        else:
            file_path = "Data/dissip_trajectories.h5"
        with h5py.File(file_path, "a") as file:
            if gamma == 0:
                worker = compute_poincare_verlet
                print("Using velocity verlet")
            else:
                worker = compute_poincare_rk
                print("Using DOP853")

            for alpha, omega, theta, p, dt, t, T, gamma in tqdm(
                    pool.imap_unordered(worker, param_list),
                    total=len(param_list),
                    desc="Computing trajectories"
                    ):

                alpha_grp = storage_setup.get_or_create_group(file, f"alpha{np.rad2deg(alpha):05.2f}", attrs={"alpha":alpha})
                omega_grp = storage_setup.get_or_create_group(alpha_grp, f"omega{omega:06.3f}", attrs={"omega":omega})
                if dt is not None:
                    init_grp = storage_setup.get_or_create_group(omega_grp, f"init{np.rad2deg(theta0):04.1f}_{p0:04.1f}", attrs={
                        "theta0": theta0,
                        "p0": p0,
                        })
                else:
                    init_grp = storage_setup.get_or_create_group(omega_grp, f"init{np.rad2deg(theta0):04.1f}_{p0:04.1f}_{gamma}", attrs={
                        "theta0": theta0,
                        "p0": p0,
                        })


                if strob:
                    init_grp.attrs["T"] = T
                if dt is None:
                    init_grp.attrs["gamma"] = gamma
                    storage_setup.create_or_overwrite_dataset(init_grp, "t", t)
                else:
                    init_grp.attrs["dt"] = dt

                storage_setup.create_or_overwrite_dataset(
                        init_grp, "theta", theta
                        )
                storage_setup.create_or_overwrite_dataset(
                        init_grp, "p", p
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
    theta0 = np.deg2rad(0.1)
    p0 = 0

    #alphas_deg = np.arange(0,15,0.01)
    alphas_deg = np.array([0])
    alphas_rad = np.deg2rad(alphas_deg)
    #omegas = [i for i in range(1,11)]
    omegas = np.arange(0.01,10, 0.01)
    alphas_omegas = [(a,w) for a in alphas_rad for w in omegas]
    param_scan(theta0, p0, alphas_omegas, strob=True)
