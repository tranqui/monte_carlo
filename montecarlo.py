#!/usr/bin/env python3
"""Monte Carlo simulation of hard spheres.

This is an example completed implementation of the Metropolis-Hastings algorithm for teaching.
This code is not intended to be given to students until after they have attempted writing their own
versions.

authors: Joshua F. Robinson <joshua.robinson@bristol.ac.uk>
             Peter Crowther <pc9836@bristol.ac.uk>

Based on code originally written by C. P. Royall.
"""

import numpy as np

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
    return np.sqrt(np.sum(dr**2))

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


class BadMonteCarlo(Exception):
    """Exception raised when hard spheres overlap.

    Used to abort a trial Monte Carlo step in the Metropolis-Hastings algorithm.
    """

class SquareWell:
    """Square well interaction potential between particles.

    This potential defines the parameters for a square well with a width and depth.
    """

    def __init__(self, depth, width):
        self.depth = depth
        self.width = width

def sweep(snap, square_well=None, step_size=0.5):
    """Single Monte Carlo sweep for hard spheres using the Metropolis-Hastings algorithm.

    Args:
        snap (snapshot): initial snapshot before sweep
        square_well: potential for interaction between hard spheres outside of contact
        step_size (float): size of trial Monte Carlo steps for each particle
    Returns:
        snap (snapshot): updated snapshot from sweep
    """
    # Some random perturbations of each particle.
    perturbation = step_size * np.random.uniform(-1,1, snap.x.shape)

    for step in range(snap.n):
        # Pick a random atom to move.
        atom = np.random.randint(snap.n)
        # Trial position for the Monte Carlo step.
        new_pos = snap.x[atom] + perturbation[step]

        # If the particle can move to the new position without overlapping others, then do it.
        try:
            # Potential energies for the square well.
            initial_energy, updated_energy = 0., 0.

            # Metropolis-Hastings rule
            for neighbour in range(snap.n):
                if neighbour is atom: continue

                # Prevent particle overlap
                dr = distance(new_pos, snap.x[neighbour], snap.box)
                overlap = (dr <= 1.)
                if overlap: raise BadMonteCarlo

                # If this is for the square well find the change in potential energy from the step.
                if square_well:
                    if dr <= square_well.width: updated_energy += square_well.depth
                    dr = distance(snap.x[atom], snap.x[neighbour], snap.box)
                    if dr <= square_well.width: initial_energy += square_well.depth

            # Metropolis-Hastings rule on the square well potential energy change.
            if square_well:
                delta = updated_energy - initial_energy
                if np.random.rand() > np.exp(-delta): raise BadMonteCarlo

            # If we did not abort, then the new position is valid.
            snap.x[atom] = apply_periodicity(new_pos, snap.box)
        # Monte Carlo acceptance rule failed, so abort the step of the particle.
        except BadMonteCarlo: pass

    snap.time += 1
    return snap


# Testing software: the student need not change the code below here - it is purely for testing
# the above code. Running this module standalone will test the main functions above.


def run_test(test, expected_result=None):
    """Evaluate an expression and see if it gives the correct result.

    A test is successful if the return value of the expression matches (up to a numerical tolerance)
    the expected value given.

    Args:
        test (str): python expression to evaluate
        expected_result: expected return value
    """
    # Run the test, get the actual result.
    print()
    print(test)
    result = eval(test)
    print('>>>', result)

    # If the result is a floating point type, allow a small deviation from the exact result due to rounding errors.
    if type(expected_result) is list:
        delta = np.array(expected_result) - np.array(result)
        success = sum(abs(delta)) < 1e-8
    elif type(expected_result) in [float, np.float, np.double, np.longdouble]:
        success = abs(expected_result-result) < 1e-8
    # Non-floating point types must be exact (e.g. integers/booleans) as they have no errors (in theory).
    else: success = expected_result == result

    if success: print('success')
    else: print('failure: expected %r' % expected_result)

# Default behaviour for the module runs a series of tests of the Monte Carlo code.
if __name__ == '__main__':
    print('<test>')
    print('>>> [your result]')
    print('[success/failure]')

    box = 2*np.ones(3)

    run_test('apply_periodicity(dr=[0.5, 0.0, 0.0], box=[2.0, 2.0, 2.0])', [ 0.5, 0.0, 0.0])
    run_test('apply_periodicity(dr=[1.0, 0.0, 0.0], box=[2.0, 2.0, 2.0])', [ 1.0, 0.0, 0.0])
    run_test('apply_periodicity(dr=[1.5, 0.0, 0.0], box=[2.0, 2.0, 2.0])', [-0.5, 0.0, 0.0])
    run_test('apply_periodicity(dr=[1.5, 1.6, 1.5], box=[2.0, 2.0, 2.0])', [-0.5,-0.4,-0.5])

    run_test('distance(a=np.zeros(3), b=np.array([0.5, 0.0, 0.0]), box=[2.0, 2.0, 2.0])', 0.5)
    run_test('distance(a=np.zeros(3), b=np.array([0.5, 0.5, 0.0]), box=[2.0, 2.0, 2.0])', np.sqrt(2*0.5**2))
    run_test('distance(a=np.zeros(3), b=np.array([0.5, 0.5, 0.0]), box=[2.0, 2.0, 2.0])', np.sqrt(2*0.5**2))
    run_test('distance(a=np.zeros(3), b=np.array([0.5, 0.5, 1.5]), box=[2.0, 2.0, 2.0])', np.sqrt(3*0.5**2))

    run_test('overlap(a=np.zeros(3), b=np.array([0.5, 0.0, 0.0]), box=[2.5, 2.5, 2.5])', True)
    run_test('overlap(a=np.zeros(3), b=np.array([0.5, 0.5, 0.0]), box=[2.5, 2.5, 2.5])', True)
    run_test('overlap(a=np.zeros(3), b=np.array([0.5, 0.5, 0.0]), box=[2.5, 2.5, 2.5])', True)
    run_test('overlap(a=np.zeros(3), b=np.array([0.5, 0.5, 1.5]), box=[2.5, 2.5, 2.5])', False)
    run_test('overlap(a=np.zeros(3), b=np.array([1.25, 0.0, 0.0]), box=[2.5, 2.5, 2.5])', False)
    run_test('overlap(a=np.zeros(3), b=np.array([2.0, 0.0, 0.0]), box=[2.5, 2.5, 2.5])', True)
