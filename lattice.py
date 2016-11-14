#!/usr/bin/env python3
"""
Module for building crystal lattices.

The module defines:
  - cell: class containing common cubic unit cells
  - create: build a lattice by translating a unit cell over a mesh

author: Joshua F. Robinson <joshua.robinson@bristol.ac.uk>
"""

import numpy as np
import sys
from atom import snapshot

class cell:
    """Unit cells forming common cubic lattices."""
    simple_cubic = np.array([0.0, 0.0, 0.0])

    BCC = np.array([[0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5]])

    FCC = np.array([[0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5]])

    octahedral_sites = np.array([[0.5, 0.0, 0.0],
                                 [0.0, 0.5, 0.0],
                                 [0.0, 0.0, 0.5],
                                 [0.5, 0.5, 0.5]])

def create(n, unit_cell):
    """Build a cubic lattice by translating a unit cell.

    >>> lattice.create([2,1,1], lattice.cell.BCC).x
    array([[ 0. ,  0. ,  0. ],
           [ 0.5,  0.5,  0.5],
           [ 1. ,  0. ,  0. ],
           [ 1.5,  0.5,  0.5]])

    Args:
        n (int array): number of unit cells in each dimension
        unit_cell (array): unit cell to be translated in each dimension
    Returns:
        snap (snapshot): the coordinates of particles in the resulting lattice
    """
    d = len(n)
    num_cells = np.prod(n)

    atoms_per_cell = len(unit_cell)
    num_atoms = num_cells*atoms_per_cell
    snap = snapshot()
    snap.x = np.tile(unit_cell, (num_cells,1))

    grid = [np.arange(n[c]) for c in range(d)]
    grid = np.array(np.meshgrid(*grid)).T
    grid = grid.reshape(-1,d)

    lattice_sites = np.repeat(grid, atoms_per_cell, 0)
    snap.x += lattice_sites

    snap.box = np.array(n, dtype=float)
    snap.types = np.array(['A']*snap.n)

    return snap

if __name__ == '__main__':
    # Default behaviour generates a lattice and displays it.
    if len(sys.argv) < 3:
        sys.stderr.write('usage: lattice.py <unit cell> [n cells array]\n')
        exit(0)

    unit_cell = cell.__dict__[sys.argv[1]]
    n = np.array(sys.argv[2:], dtype=np.int)
    d = unit_cell.shape[1]
    if len(n) is not d:
        sys.stderr.write('error: number of unit cells n=%r does not match dimension of lattice d=%r\n' % (n, d))
        exit(0)

    # Create the lattice.
    lattice = create(n, unit_cell)
    print(lattice)
