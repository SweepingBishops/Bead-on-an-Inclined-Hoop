#!/usr/bin/env python
"""
Poincaré and Trajectory Data Storage (HDF5)
=========================================

Overview
--------
This project stores numerical trajectories and Poincaré sections of a driven
nonlinear dynamical system in HDF5 (.h5) format using `h5py`.

The data format is designed to be:
- hierarchical and self-describing
- memory-safe via lazy loading
- suitable for large parameter scans
- reusable by users who did not write the original code

Typical files produced by this project are:

    trajectories.h5                  Full trajectories (Hamiltonian)
    poincare_trajectories.h5          Poincaré sections (Hamiltonian)
    dissip_trajectories.h5            Full trajectories (dissipative)
    dissip_poincare_trajectories.h5   Poincaré sections (dissipative)


File Structure
--------------
Each HDF5 file follows the same logical hierarchy:

    <file>.h5
    ├── (file attributes)
    │   ├── integrator
    │   ├── data_type
    │   ├── dt
    │   ├── dt_units
    │   ├── alpha_units
    │   ├── omega_units
    │
    ├── alpha0
    │   ├── (attrs: alpha)
    │   ├── omega0
    │   │   ├── (attrs: omega)
    │   │   ├── init_0
    │   │   │   ├── theta   (dataset)
    │   │   │   └── p       (dataset)
    │   │   └── init_1
    │   └── omega1
    │
    └── alpha1

Hierarchy semantics:
- alphai   : one value of the control parameter alpha
- omegaj   : one value of the drive frequency omega
- init_k    : one initial condition
- theta, p  : phase-space data

Physical parameter values are stored as ATTRIBUTES, not encoded in group names.


Metadata
--------
Important metadata are stored as HDF5 attributes.

At the file level:
- integrator   : numerical integrator used (e.g. "velocity_verlet")
- data_type    : description of stored data
- dt           : integration time step
- *_units      : physical units of parameters

At the group level:
- alpha        : value of alpha
- omega        : value of omega
- theta0, p0   : initial conditions (for init_k groups)


Basic Usage
-----------
Inspect file-level metadata:

    >>> import h5py
    >>> with h5py.File("poincare_trajectories.h5", "r") as f:
    ...     print(f.attrs["integrator"])
    ...     print(f.attrs["data_type"])

List available alpha values:

    >>> with h5py.File("poincare_trajectories.h5", "r") as f:
    ...     for key in f:
    ...         print(f[key].attrs["alpha"])

Load a single Poincaré section:

    >>> with h5py.File("poincare_trajectories.h5", "r") as f:
    ...     grp = f["alpha_0/omega_3/init_0"]
    ...     theta = grp["theta"][:]
    ...     p = grp["p"][:]

Only the requested datasets are loaded into memory.


Safe Iteration Pattern
----------------------
To loop over all stored data without exhausting memory:

    >>> with h5py.File("poincare_trajectories.h5", "r") as f:
    ...     for grp_alpha in f.values():
    ...         alpha = grp_alpha.attrs["alpha"]
    ...         for grp_omega in grp_alpha.values():
    ...             omega = grp_omega.attrs["omega"]
    ...             for grp_init in grp_omega.values():
    ...                 theta = grp_init["theta"][:]
    ...                 p = grp_init["p"][:]
    ...                 # analysis here


Important Notes
---------------
- Do NOT load entire files into memory at once.
- Always rely on attributes for physical parameter values.
- Group names (alpha_0, omega_3, etc.) are identifiers, not values.
- Use lazy loading (dataset[:]) only when data are needed.
"""

import os.path
import numpy as np
import h5py

def get_or_create_group(parent, name, attrs=None):
    """
    Gets or creates a group.
    parent: a h5 file or group
    name: string that sets the name of the group
    attrs: dictionary of attributes to be added to the group
    """
    if name in parent:
        group = parent[name]
        if attrs:
            for k,v in attrs.items():
                group.attrs[k] = v
        return group
    else:
        group = parent.create_group(name)
        for k,v in parent.attrs.items():
            group.attrs.setdefault(k,v)
        if attrs:
            for k,v in attrs.items():
                group.attrs[k] = v
    return group

def create_or_overwrite_dataset(parent, name, data, attrs=None):
    """
    Creates a dataset and overwrites it if it exists.
    parent: h5 group
    name: name of the dataset (string)
    data: numpy array to put in the dataset
    attrs: dictionary of attributes to be added to the dataset
    """
    if name in parent:
        del parent[name]

    ds = parent.create_dataset(name, data=data, compression="gzip", chunks=True)

    for k,v in parent.attrs.items():
        ds.attrs.setdefault(k,v)
    if attrs:
        for k,v in attrs.items():
            ds.attrs[k] = v
        
    return ds


def setup_file(path, integrator, data_type, alphas, omegas, dt=0.05):
    if os.path.isfile(path):
        print(f"{path} already exists. Skipping...")
        return

    print(f"Setting up {path}")
    with h5py.File(path, "w") as file:
        file.attrs["integrator"] = integrator
        file.attrs["data_type"] = data_type
        file.attrs["dt"] = dt
        file.attrs["dt_units"] = "seconds"
        file.attrs["alpha_units"] = "radians"
        file.attrs["omega_units"] = "rad/s"
        for alpha in alphas:
            group = get_or_create_group(file, f"alpha{alpha:05.2f}", attrs={"alpha": np.deg2rad(alpha)})
            for omega in omegas:
                grp = get_or_create_group(group, f"omega{omega:06.3f}", attrs={"omega": omega, "T": 2*np.pi/omega if omega != 0 else None})

if __name__ == "__main__":
    alphas = [i for i in range(1,16)]
    omegas = [i for i in range(1,11)]

    setup_file("Data/trajectories.h5", "velocity_verlet", "Full trajectories, non-dissipative", alphas, omegas)
    setup_file("Data/poincare_trajectories.h5", "velocity_verlet", "Poincare trajectories sampled at t = nT, non-dissipative", alphas, omegas)
    setup_file("Data/dissip_trajectories.h5", "DOP853", "Full trajectories, dissipative", alphas, omegas)
    setup_file("Data/dissip_poincare_trajectories.h5", "DOP853", "Poincare trajectories sampled at t = nT, dissipative", alphas, omegas)


    print("Done!")
