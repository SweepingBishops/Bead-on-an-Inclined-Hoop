''' Code to generate the bifurcation diagrams -
             a) Fixed point theta^* as a function of the driving frequency omega
             b) Fixed point theta^* as a function of the driving amplitude alpha
             c) Fixed point theta^* as a function of time t
    For each plot, we keep the other parameters fixed and vary the parameter of interest.
'''

import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
        

# ----- Fixed Point Criterion -----
def f(theta, omega, t, g, alpha):
    return ((np.cos(alpha)*g - omega**2 * np.cos(theta)) * np.sin(theta)
            - (g) * np.sin(alpha) * np.cos(omega*t) * np.cos(theta))


def f_theta(theta, omega, t,g,alpha):
    A = np.cos(alpha)*g
    B = (g)*np.sin(alpha)*np.cos(omega*t)

    return (
        A*np.cos(theta)
        - omega**2 * np.cos(2*theta)
        + B*np.sin(theta)
    )


def classify_root(theta_star, omega, t, b=0, g=1, alpha=0):
    d = f_theta(theta_star, omega, t , g, alpha)

    if d > 0:
        return "stable" if b > 0 else "center"
    else:
        return "unstable"
    
def find_all_roots_with_stability(omega, t, b=0, g=1, alpha=0, n_points=1500):

    thetas = np.linspace(-np.pi, np.pi, n_points)
    values = f(thetas, omega, t, g, alpha)

    roots = []

    for i in range(len(thetas)-1):
        if values[i] == 0:
            roots.append(thetas[i])
        elif values[i]*values[i+1] < 0:
            try:
                sol = root_scalar(
                    f,
                    args=(omega, t, g, alpha),
                    bracket=[thetas[i], thetas[i+1]],
                    method='brentq'
                )

                r = np.round(sol.root, 6)
                stab = classify_root(r, omega, t, b, g, alpha)

                roots.append((r, stab))

            except:
                pass

    # remove duplicates
    unique = {}
    for r, s in roots:
        unique[r] = s

    return [(r, unique[r]) for r in sorted(unique)]

def plot_bifurcation_omega(omega_range, t, alpha, g =1, b=0):
    bifurcation_data = []

    for omega in omega_range:
        roots = find_all_roots_with_stability(omega, t, b, g, alpha)
        for r, s in roots:
            bifurcation_data.append((omega, r, s))
    bifurcation_data = np.array(bifurcation_data)

    plt.figure(figsize=(10, 6))
    for omega, r, s in bifurcation_data:
        color = 'blue' if s == 'stable' else 'red' if s == 'unstable' else 'green'
        plt.scatter(omega, r, color=color, s=10)
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\theta^*$')
    plt.title('Bifurcation Diagram: Fixed Points vs Driving Frequency')
    plt.grid()
    plt.show()

def plot_bifurcation_alpha(alpha_range, t, omega, g=1, b=0):
    bifurcation_data = []

    for alpha in alpha_range:
        roots = find_all_roots_with_stability(omega, t, b, g, alpha )
        for r, s in roots:
            bifurcation_data.append((alpha, r, s))
    bifurcation_data = np.array(bifurcation_data)

    plt.figure(figsize=(10, 6))
    for alpha, r, s in bifurcation_data:
        color = 'blue' if s == 'stable' else 'red' if s == 'unstable' else 'green'
        plt.scatter(alpha, r, color=color, s=10)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\theta^*$')
    plt.title('Bifurcation Diagram: Fixed Points vs Driving Amplitude')
    plt.grid()
    plt.show()

def plot_bifurcation_time(t_range, omega, alpha, g=1, b=0):
    bifurcation_data = []

    for t in t_range:
        roots = find_all_roots_with_stability(omega, t, b, g, alpha)
        for r, s in roots:
            bifurcation_data.append((t, r, s))
    bifurcation_data = np.array(bifurcation_data)

    plt.figure(figsize=(10, 6))
    for t, r, s in bifurcation_data:
        color = 'blue' if s == 'stable' else 'red' if s == 'unstable' else 'green'
        plt.scatter(t, r, color=color, s=10)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\theta^*$')
    plt.title('Bifurcation Diagram: Fixed Points vs Time')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    omega_range = np.linspace(0.5, 10.0, 50)
    t = 0
    alpha = np.radians(0)
    plot_bifurcation_omega(omega_range, t, alpha)

    alpha_range = np.linspace(0, np.radians(60), 50)
    omega = 2.0
    plot_bifurcation_alpha(alpha_range, t, omega)

    t_range = np.linspace(0, 20*np.pi, 100)
    alpha = np.radians(16)
    omega = np.sqrt(np.cos(alpha))
    plot_bifurcation_time(t_range, omega, alpha)

        

