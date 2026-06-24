# Bead on an Inclined Hoop
Numerical study of the bead on an inclined rotating hoop - study of dynamics, bifurcation and chaos.

## Usage
On first use run `storage_setup.py` from the root of this repository. It will create four `.h5` files in the
`./Data` directory. This will be where all the trajectories will be stored.

Each HDF5 file follows the same logical hierarchy:

```

    <file>.h5
    ├── (file attributes)
    │   ├── integrator
    │   ├── data_type
    │   ├── dt
    │   ├── dt_units
    │   ├── alpha_units
    │   └── omega_units
    │
    ├── alpha00.00
    │   ├── (attrs: alpha)
    │   ├── omega00.000
    │   │   ├── (attrs: omega)
    │   │   ├── strob00.0_00.0
    │   │   │   ├── theta   (dataset)
    │   │   │   ├── thetadot(dataset)
    │   │   │   └── tau     (dataset)
    │   │   ├── uniform00.0_00.0
    │   │   │   ├── theta   (dataset)
    │   │   │   ├── thetadot(dataset)
    │   │   │   └── tau     (dataset)
    │   │   └── strob01.0_00.0
    │   └── omega01.000
    │
    └── alpha01.00
```

See the documentation in `storage_setup.py` for more details.

The `param_scan` function from `parameter_scan.py` is what actually solves all the trajectories.
It is recommended to use the helper script in `solve_trajectories.py`, just change the parameter values as required.

This is outdated:
> Note that stroboscopic trajectores are written to `*poincare_trajectories.h5` and trajectories with a damping are
> written to `dissip*`.

All trajectories are now uniformly time-stepped. Use the `samples_per_period` attribute of the trajectory group to
construct a stroboscopic/Poincaré maps. For example: if `samples_per_period`=64, use `dataset[::64]`.

For drawing plots use `phase_trajectory_plot.py`, `poincare_plots.py`,
`strob_plot_alpha.py`, and `strob_plot_omega.py`. There will be list of alpha and omega values to plot
for at the beginning of each file. Edit those as necessary.
