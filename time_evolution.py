#!/usr/bin/env python
# Code to generate the time evolution of the system 

import numpy as np
from scipy.integrate import solve_ivp
from math import sin, cos


#=========================================================
# Functions to generate the time evolution of the system
#=========================================================

# def force(theta, t,
#           omega, alpha,
#           g_eff = 9.8/0.1775  # g/hoop radius
#           ):
#     """
#     Used for velocity verlet integration.
#     Hamiltonian force: dp/dt = -∂V/∂θ
#     """
#     return (
#         - (np.cos(alpha) * g_eff - omega**2 * np.cos(theta)) * np.sin(theta)
#         + g_eff * np.sin(alpha) * np.sin(omega * t) * np.cos(theta)
#     )
# 
# 
# def velocity_verlet(theta0, p0, t_fin,
#                     omega, alpha,
#                     discard_initial_time=0,  # Initial transient time to discard from result (in sec)
#                     t_in=0, dt=0.01,
#                     strob=False,  # whether to sample stroboscopically
#                     strob_time=1  # If strob=True, then time period of sampling. Note strob time is taken after
#                                   # the initial steps are discarded according to discard_initial_time
#                     ):
#     '''Symplectic Integrator. !!DO NOT USE WITH DAMPING!!'''
# 
#     n_steps = int( (t_fin-t_in) /dt)
#     discard_steps = int(discard_initial_time/dt)
#     if discard_steps > n_steps:
#         raise ValueError("Discard time larger than integration time")
# 
#     theta = np.empty(n_steps)
#     p = np.empty(n_steps)
#     t = np.empty(n_steps)
# 
#     theta[0] = theta0
#     p[0] = p0
#     t[0] = t_in
# 
#     for n in range(n_steps - 1):
#             # half step momentum
#             p_half = p[n] + 0.5 * dt * force(theta[n], t[n], omega, alpha)
# 
#             # full step position
#             theta[n+1] = theta[n] + dt * p_half
#             t[n+1] = t[n] + dt
# 
#             # second half step momentum
#             p[n+1] = p_half + 0.5 * dt * force(theta[n+1], t[n+1], omega, alpha)
# 
#     if not strob:
#         return t[discard_steps:], theta[discard_steps:], p[discard_steps:]
#     else:
#         theta = theta[discard_steps:]
#         p = p[discard_steps:]
#         t = t[discard_steps:]
# 
#         T = strob_time
#         nT = np.round(t / T).astype(int)
#         mask = np.abs(t - nT*T) < dt/2
# 
#         theta_strob = theta[mask]
#         p_strob = p[mask]
#         t_strob = t[mask]
#         return t_strob, theta_strob, p_strob


def time_evolve_rk(
    theta0,
    thetadot0,
    tau_fin,  # omega*t_fin
    alpha,
    A,  # g/(R* omega**2)
    B,  # ɣ/omega
    discard_tau=0,  # initial transient time to discard from results (in tau)
    gamma=0.5,
    method="DOP853",   #DOP853 is a high-order explicit Runge-Kutta method - RK45 might induce damping artifacts for long time evolutions even when b = 0.
    tau_in=0,  # omega*t_in
    rtol=1e-7,
    atol=1e-8,
    ):
    """
    Time evolve the driven pendulum with damping.

    Returns:
        tau_vals, theta_vals, theta_dot_vals
    """

    A_cos_alpha = A*cos(alpha)
    A_sin_alpha = A*sin(alpha)

    def dynamical_system(tau, config):
        theta, theta_dot = config

        dtheta_dtau = theta_dot

        dtheta_dot_dtau = (
            - B * theta_dot
            - (A_cos_alpha - cos(theta)) * sin(theta)
            + A_sin_alpha * np.sin(tau) * np.cos(theta)
        )

        return [dtheta_dtau, dtheta_dot_dtau]

    tau_span = (tau_in, tau_fin)

    y0 = [theta0, thetadot0]

    sol = solve_ivp(
        dynamical_system,
        tau_span,
        y0,
        method=method,
        dense_output=True,
        rtol=rtol,
        atol=atol,
    )

    samples_per_period = 128

    T = 2*np.pi

    tau_uniform = np.arange(
            discard_tau,
            tau_fin,
            T/samples_per_period
            )

    tau_strob = np.arange(discard_tau, tau_fin, T)

    y_uniform = sol.sol(tau_uniform)
    y_strob = sol.sol(tau_strob)

    return (tau_uniform, y_uniform), (tau_strob, y_strob)



#-------------------------
# Example usage
#-------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    g = 9.8
    R = 0.355/2
    omega = 1.34
    gamma = 0.5

    A = g/(R * omega**2)
    B = gamma/omega

    uniform, strob = time_evolve_rk(
        theta0=0.1,
        thetadot0=0.0,
        tau_fin = 1000*2*np.pi,
        discard_tau=100*2*np.pi,
        A=A,
        B=B,
        alpha=np.deg2rad(60),
    )

    plt.figure()
    plt.plot(uniform[0], uniform[1][0])
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\theta(\tau)$")
    plt.grid(True)

    # plt.figure()
    # plt.plot(theta[:], theta_dot[:], lw=0.2, alpha=0.5)
    # plt.xlabel(r"$\theta$")
    # plt.ylabel(r"$\dot{\theta}$")
    # plt.grid(True)

    plt.show()
