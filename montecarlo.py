#!/usr/bin/env python3
"""Monte Carlo simulation of hard spheres.

This is an example completed implementation of the Metropolis-Hastings algorithm for teaching.
This code is not intended to be given to students until after they have attempted writing their own
versions.

authors: Joshua F. Robinson <joshua.robinson@bristol.ac.uk>
             Peter Crowther <pc9836.2010@my.bristol.ac.uk>

Based on code originally written by C. P. Royall.
"""

import numpy as np
from numpy.random import uniform

def apply_periodicity(dr, box):
    """Apply periodic boundary conditions to the displacement vector.

    This retrieves the shortest vector of displacement between two particles, after taking
    into account their images due to the periodic boundaries.

    Args:
        dr (array): the unwrapped displacement vector between the particles (i.e. before applying boundary conditions)
        box (array): the box size in each dimension
    Returns:
        dr (array): the wrapped displacement vector (i.e. after applying boundary conditions)
    """
    for c in range(len(dr)):
        if dr[c] < -0.5*box[c]: dr[c] += box[c]
        if dr[c] >  0.5*box[c]: dr[c] -= box[c]
    return dr

def distance(a, b, box):
    """Compute distance between two particles within a periodic box.

    Args:
        a (array): position vector of first particle
        b (array): position vector of second particle
        box (array): size of box in each dimension
    Returns:
        distance (float): shortest distance between two particles or their images
    """
    dr = b-a
    dr = apply_periodicity(dr, box)
    return np.sqrt(sum(dr**2))

def overlap(a, b, box):
    """Determine whether two hard spheres overlap.

    Args:
        a (array): position vector of first particle
        b (array): position vector of second particle
        box (array): size of box in each dimension
    Returns:
        overlap (bool): whether min(|b-a|) <= hard sphere diameter, where min(...) acts over the particles and their images (i.e. we apply periodic boundary conditions from the box)
    """
    dr = distance(a, b, box)
    return dr <= 1.


class OverlapError(Exception):
    """Exception raised when hard spheres overlap.

    Used to abort a trial Monte Carlo step in the Metropolis-Hastings algorithm.
    """

def sweep(current, step_size=0.1):
    """Single Monte Carlo sweep for hard spheres using the Metropolis-Hastings algorithm.

    Args:
        current (snapshot): initial snapshot before sweep
        step_size (float): size of trial Monte Carlo steps for each particle
    Returns:
        current (snapshot): updated snapshot from sweep
    """
    # Some random perturbations of each particle.
    perturbation = step_size*uniform(-1,1, current.x.shape)

    for atom in range(current.n):
        # Trial position for the Monte Carlo step.
        new_pos = current.x[atom] + perturbation[atom]

        # If the particle can move to the new position without overlapping others, then do it.
        try:
            # Metropolis-Hastings rule: prevent particle overlap
            for neighbour in range(current.n):
                if neighbour is atom: continue
                if overlap(new_pos, current.x[neighbour], current.box): raise OverlapError
            # If we did not abort, then the new position is valid.
            current.x[atom] = new_pos
        # Particles overlapped, so abort the step of the particle.
        except OverlapError: pass

    current.time += 1
    return current
