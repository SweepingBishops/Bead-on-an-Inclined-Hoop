# Code to generate the time evolution of the system 

import numpy as np
from scipy.integrate import solve_ivp


################################################################
# Function to generate the time evolution of the system
################################################################

def time_evolve(
    theta0,
    theta_dot0,
    t_fin,
    b=0.0,
    m=1.0,
    g=1.0,
    R0=1.0,
    alpha=np.pi/180 * 16,
    omega=None,
    t_in=0.0,
    n_points=50000,
    method="DOP853"   #DOP853 is a high-order explicit Runge-Kutta method - RK45 might induce damping artifacts for long time evolutions even when b = 0.
):
    """
    Time evolve the driven pendulum with damping.

    Returns:
        t_vals, theta_vals, theta_dot_vals
    """

    if omega is None:
        omega = np.sqrt(np.cos(alpha) * g / R0)

    def dynamical_system(t, y):
        theta, theta_dot = y

        dtheta_dt = theta_dot

        dtheta_dot_dt = (
            - (np.cos(alpha) * g / R0 - omega**2 * np.cos(theta)) * np.sin(theta)
            + (g / R0) * np.sin(alpha) * np.cos(omega * t) * np.cos(theta)
            - (b / m) * theta_dot
        )

        return [dtheta_dt, dtheta_dot_dt]

    t_span = (t_in, t_fin)
    t_eval = np.linspace(t_in, t_fin, n_points)
    y0 = [theta0, theta_dot0]

    sol = solve_ivp(
        dynamical_system,
        t_span,
        y0,
        t_eval=t_eval,
        method=method
    )

    return sol.t, sol.y[0], sol.y[1]


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t, theta, theta_dot = time_evolve(
        theta0=0.1,
        theta_dot0=0.0,
        t_fin=10000,
        b=0.1,
        n_points=100000
    )

    plt.figure()
    plt.plot(t, theta)
    plt.xlabel("t")
    plt.ylabel(r"$\theta(t)$")
    plt.grid(True)

    plt.figure()
    plt.plot(theta[:], theta_dot[:])
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.grid(True)

    plt.show()
