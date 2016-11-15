#!/usr/bin/env python3
"""
Module for computing pair distribution functions from snapshots.

This module uses advanced python features & libraries, and as such the student is not
intended to change the code. A curious student may wish to examine the code out of interest.

The module defines:
  - unit_sphere_volume: volume of a sphere of unit diameter
  - bin_distances: histograms the pair distances within a snapshot
  - pair_distribution: computes the g(r) from a given trajectory

author: Joshua F. Robinson <joshua.robinson@bristol.ac.uk>
"""

import numpy as np
from scipy.special import factorial

def unit_sphere_volume(d):
    """Volume of a unit sphere in d-dimensions.

    Used for normalisation of the pair distribution.

    Args:
        d (int): dimension of the sphere
    Returns:
        volume (float): volume of the hypersphere of radius 1."""
    return np.pi**(d/2) / factorial(d/2)

def bin_distances(snap, bins):
    """Compute a histogram of distances between particles in a snapshot.

    Args:
        snap (snapshot): configuration to compute distances from
        bins (array): bins for the histogram
    Returns:
        counts (array): number of distances which fall within each histogram bin
    """
    d = snap.d
    n = snap.n
    box = snap.box
    coords = snap.x.copy()

    # Distance between every particle
    rij = coords[:,None] - coords[None,:]
    # Apply periodic boundary conditions
    for c in range(d):
        rij[rij[:,:,c] < -0.5*box[c], c] += box[c]
        rij[rij[:,:,c] >  0.5*box[c], c] -= box[c]

    dists = np.sqrt((rij**2.0).sum(axis=2))
    dists = dists[np.triu_indices(n,1)]
    counts, _ = np.histogram(dists, bins, density=False)

    # Return double the counts, because each pair distance involves two particles
    return 2*counts

def pair_distribution(trajectory, num_bins=25):
    """Compute g(r) averaged over every snapshot in the given trajectory.

    Args:
        trajectory (list or snapshot): configurations over which to average the g(r)
        num_bins (int): number of bins for the g(r)
    Returns:
        r (array): distances in the g(r)
        g (array): g(r) at every distance r
    """
    if type(trajectory) is not list: trajectory = [trajectory]

    # 
    box = trajectory[0].box
    bins = np.linspace(0, 0.5*min(box), num_bins+1)

    counts = np.zeros(num_bins)
    for snap in trajectory: counts += bin_distances(snap, bins)
    counts *= 1./(len(trajectory))

    d = snap.d
    shell_volume = unit_sphere_volume(d) * (bins[1:]**d - bins[:-1]**d)
    total_volume = np.prod(box)
    density = snap.n / total_volume
    # Normalise by the ideal gas expectation (i.e. when there is no correlation).
    g = counts / (snap.n*density*shell_volume)

    r = 0.5*(bins[1:] + bins[:-1])
    return r, g

if __name__ == '__main__':
    # Get plotting tools.
    try: import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write('cannot import plotting software to test g(r)!')
        exit(0)

    # Create an ideal gas by randomly distributing particles.
    from atom import snapshot
    n, d = 1000, 3
    snap = snapshot(np.random.random((n,d)), box = np.ones(d), types = ['A']*n)
    r, g = pair_distribution(snap)

    # Plot the result: it should be g(r)=1 for all r.
    plt.plot(r, g)
    plt.xlabel('r')
    plt.ylabel('g(r)')
    # Expected result for reference.
    plt.axhline(y=1, linestyle='dashed', color='black')

    # There are numerical fluctuations at small r because of the normalisation by the volume,
    # which becomes tiny at small r amplifying numerical errors. We only plot for larger r to
    # circumvent this.
    plt.xlim([0.1, 0.5])

    plt.show()
