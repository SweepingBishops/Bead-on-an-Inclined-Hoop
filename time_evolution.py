#!/usr/bin/env python
# Code to generate the time evolution of the system 

import numpy as np
from scipy.integrate import solve_ivp


#=========================================================
# Functions to generate the time evolution of the system
#=========================================================

def force(theta, t,
          omega, alpha,
          g_eff = 9.8/0.355  # g/hoop radius
          ):
    """
    Used for velocity verlet integration.
    Hamiltonian force: dp/dt = -∂V/∂θ
    """
    return (
        - (np.cos(alpha) * g_eff - omega**2 * np.cos(theta)) * np.sin(theta)
        + g_eff * np.sin(alpha) * np.sin(omega * t) * np.cos(theta)
    )


def velocity_verlet(theta0, p0, t_fin,
                    omega, alpha,
                    discard_initial_time=0,  # Initial transient time to discard from result (in sec)
                    t_in=0, dt=0.05,
                    strob=False,  # whether to sample stroboscopically
                    strob_time=1  # If strob=True, then time period of sampling. Note strob time is taken after
                                  # the initial steps are discarded according to discard_initial_time
                    ):
    '''Symplectic Integrator. !!DO NOT USE WITH DAMPING!!'''

    n_steps = int( (t_fin-t_in) /dt)
    discard_steps = int(discard_initial_time/dt)

    theta = np.empty(n_steps)
    p = np.empty(n_steps)
    t = np.empty(n_steps)

    theta[0] = theta0
    p[0] = p0
    t[0] = t_in

    for n in range(n_steps - 1):
            # half step momentum
            p_half = p[n] + 0.5 * dt * force(theta[n], t[n], omega, alpha)

            # full step position
            theta[n+1] = theta[n] + dt * p_half
            t[n+1] = t[n] + dt

            # second half step momentum
            p[n+1] = p_half + 0.5 * dt * force(theta[n+1], t[n+1], omega, alpha)

    if not strob:
        return t[discard_steps:], theta[discard_steps:], p[discard_steps:]
    else:
        theta = theta[discard_steps:]
        p = p[discard_steps:]
        t = t[discard_steps:]

        T = strob_time
        nT = np.round(t / T).astype(int)
        mask = np.abs(t - nT*T) < dt/2

        theta_strob = theta[mask]
        p_strob = p[mask]
        t_strob = t[mask]
        return t_strob, theta_strob, p_strob


def time_evolve_rk(
    theta0,
    theta_dot0,
    t_fin,
    omega,
    alpha,
    discard_initial_time=0,  # initial transient time to discard from results (in sec)
    gamma=0,
    g_eff = 9.8/0.355,  # g/hoop radius
    t_in=0,
    n_points=50000,
    method="DOP853",   #DOP853 is a high-order explicit Runge-Kutta method - RK45 might induce damping artifacts for long time evolutions even when b = 0.
    strob=False,   # Whether to sample stroboscopically
    strob_time=1  # If strob=True, then time period for sampling. Note strob time is taken after
                  # the initial steps are discarded according to discard_initial_time
    ):
    """
    Time evolve the driven pendulum with damping.

    Returns:
        t_vals, theta_vals, theta_dot_vals
    """

    def dynamical_system(t, config):
        theta, theta_dot = config

        dtheta_dt = theta_dot

        dtheta_dot_dt = (
            - gamma * theta_dot
            - (np.cos(alpha) * g_eff - omega**2 * np.cos(theta)) * np.sin(theta)
            + g_eff * np.sin(alpha) * np.sin(omega * t) * np.cos(theta)
        )

        return [dtheta_dt, dtheta_dot_dt]

    t_span = (t_in, t_fin)
    t_start = t_in + discard_initial_time
    if strob is False:
        t_eval = np.linspace(t_start, t_fin, n_points)
    else:
        t_eval = np.arange(t_start, t_fin, strob_time)

    y0 = [theta0, theta_dot0]

    sol = solve_ivp(
        dynamical_system,
        t_span,
        y0,
        t_eval=t_eval,
        method=method
    )

    return sol.t, sol.y[0], sol.y[1]


#-------------------------
# Example usage
#-------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t, theta, theta_dot = time_evolve_rk(
        theta0=0.1,
        theta_dot0=0.0,
        t_fin=2000,
        discard_initial_time=1000,
        omega=6,
        alpha=np.deg2rad(6),
        gamma=0.1,
    )

    plt.figure()
    plt.plot(t, theta)
    plt.xlabel("t")
    plt.ylabel(r"$\theta(t)$")
    plt.grid(True)

    plt.figure()
    plt.plot(theta[:], theta_dot[:], lw=0.2, alpha=0.5)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.grid(True)


    g_eff = 9.8/0.355
    alpha = np.deg2rad(18)
    omega = np.sqrt(np.cos(alpha)*g_eff)

    t, theta, p = velocity_verlet(
            0.01, 0, 10_000, omega, alpha, strob=True, strob_time=2*np.pi/omega,
            discard_initial_time=1000
            )

    plt.figure()
    plt.scatter(theta, p, s=0.4)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.grid(True)

    plt.show()
