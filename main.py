#!/usr/bin/env python3
"""
Central script for monte carlo simulation of hard spheres and square wells.

This script brings together all the elements in other modules to run the algorithm.
The process follows:
  1. Initialise the system from a crystal lattice.
  2. Run the monte carlo for enough sweeps to melt the crystal and equilibrate.
  3. Run additional sweeps to collect uncorrelated snapshots for analysis: build a trajectory
       from these snapshots.
  4. Dump the trajectory to a file.
  5. Run analysis: compute pair distribution function g(r) averaged over the trajectory.
  6. Display analysis.

authors: Joshua F. Robinson <joshua.robinson@bristol.ac.uk>
             Peter Crowther <pc9836.2010@my.bristol.ac.uk>

Based on code originally written by C. P. Royall.
"""

import numpy as np
from atom import snapshot
from g import pair_distribution
import lattice, montecarlo, atom

# Properties of the fluid
ncells = 3              # number of initial lattice cells to create (sets the number of particles)
d = 3                   # dimension
volume_fraction = 0.3   # occupied volume: sets the density
diameter = 1.0          # diameter of hard spheres

# Settings for the simulation
num_sweeps_to_equilibrate = 25
num_sweeps_between_collects = 25
num_collections = 10

# Create the initial crystal lattice. This will create a ncells^d unit cells (of length 1 each)
unit_cell = lattice.cell.FCC
current = lattice.create([ncells]*d, unit_cell)

# Rescale the box/coordinates so we achieve the correct volume fraction/density.
atoms_per_cell = len(unit_cell)
atom_volume = np.pi*diameter**3 / 6
initial_volume_fraction = atoms_per_cell * atom_volume
rescale = (initial_volume_fraction/volume_fraction)**(1./3)
current.box *= rescale
current.x *= rescale

# Melt the crystal.
print('melting crystal...')
for sweep in range(num_sweeps_to_equilibrate): current = montecarlo.sweep(current)

# Collection run.
trajectory = [current.copy()]
print('collecting data...')
for collect in range(num_collections):
    for sweep in range(num_sweeps_between_collects): current = montecarlo.sweep(current)
    trajectory += [current.copy()]

# Dump the trajectory to a file. This can be viewed with external software e.g. ovito.
atom.write('trajectory.atom', trajectory)

# Get the pair distribution, i.e. the g(r), from the trajectory.
r, g = pair_distribution(trajectory)

print('pair distribution function:')
print('    r \t\t g(r)')
print(np.array([r, g]).transpose())

# Attempt to plot the g(r) with matplotlib.
try:
    import matplotlib.pyplot as plt
    plt.plot(r,g)
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.show()
# If matplotlib is not available, then dump the data to a file: first column r, second column g(r).
except ImportError:
    table = np.array([r,g]).transpose()
    np.savetxt('g.csv', table)
