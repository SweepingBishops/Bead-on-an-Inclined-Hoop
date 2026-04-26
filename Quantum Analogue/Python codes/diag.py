import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# parameters
t2 = 1.0
t1 = -4*t2-1
U  = 0.02

Nk = 400
k = np.linspace(-np.pi, np.pi, Nk)
dk = k[1] - k[0]

# dispersion
Ek = -2*t1*np.cos(k) - 2*t2*np.cos(2*k)

# Hamiltonian matrix
H = np.zeros((Nk,Nk))

# potential term
for i in range(Nk):
    H[i,i] = Ek[i]

# kinetic term (-U d^2/dk^2)
for i in range(1,Nk-1):
    H[i,i] += 2*U/dk**2
    H[i,i-1] += -U/dk**2
    H[i,i+1] += -U/dk**2

# diagonalize
E, V = eigh(H)

# plot dispersion and ground state
plt.figure()
plt.plot(k, Ek)
plt.title("Momentum-space potential E(k)")
plt.xlabel("k")
plt.ylabel("E(k)")
plt.show()

plt.figure()
plt.plot(k, V[:,0]**2)
plt.title("Ground state in momentum space")
plt.xlabel("k")
plt.ylabel("|psi(k)|^2")
plt.show()