#!/usr/bin/env python3
"""
Module for reading and writing snapshots from LAMMPS (.atom) file formats.

This module uses advanced python features & libraries, and as such the student is not
intended to change the code. A curious student may wish to examine the code out of interest.

The module defines:
  - snapshot: the class containing the snapshot data and handles the reading/writing
  - NoSnapshotError: exception raised when a snapshot could not be read from the file
  - write: helper function for writing a trajectory to a file
  - read: helper function for reading a trajectory from a file

author: Joshua F. Robinson <joshua.robinson@bristol.ac.uk>
"""

import sys, os, io, pandas
import numpy as np

class NoSnapshotError(RuntimeError):
    """Exception raised when not able to read a snapshot from a file."""

class snapshot:
    """Snapshot of a system of particles.

    Variables:
        n (int): number of particles in the configuration
        d (int): dimensionality of configuration space
        x (numpy array): coordinates of the particles (n by d vector)
        box (numpy array): box containing the particles
        types (str array): labels of the particle types
        time (int/float): time or frame of the snapshot within a trajectory
    """

    def __init__(self, x=np.empty((0,0)), box=None, types=None, time=0):
        """Create a new snapshot.

        Args:
            x (numpy array): coordinates of the particles (n by d vector)
            box (numpy array): box containing the particles
            types (str array): labels of the particle types
            time (int/float): time or frame of the snapshot within a trajectory
        """
        self.x = x
        self.box = box
        self.types = types
        self.time = time

    def copy(self):
        """Return a deep copy of the snapshot."""
        return snapshot(self.x.copy(), self.box.copy(), self.types.copy(), self.time)

    @property
    def n(self):
        """Number of particles in snapshot."""
        return len(self.x)

    @property
    def d(self):
        """Dimensionality of configuration space."""
        return self.x.shape[1]

    @staticmethod
    def from_file(f):
        """Load a new snapshot from a file.

        Args:
            f (file or str): file to read from
        Returns:
            snap (snapshot): the new snapshot
        Raises:
            NoSnapshotError: if could not read from file
            RuntimeException: if did not recognise file format
        """
        snap = snapshot()
        snap.read(f)
        return snap

    def read(self, f):
        """Read a snapshot from a file, overwriting any existing data.

        Args:
            f (file or str): file to read from
        Raises:
            NoSnapshotError: if could not read from file
            RuntimeException: if did not recognise file format
        """
        if type(f) is str: f = open(f)

        self.x = np.empty((0,0))
        self.time = self.box = None
        while True:
            item = f.readline().split()
            if not item: raise NoSnapshotError
            assert item[0] == 'ITEM:'

            # Timestep within a trajectory.
            if item[1] == 'TIMESTEP':
                self.time = int(f.readline())

            # Number of atoms in the header
            elif ' '.join(item[1:4]) == 'NUMBER OF ATOMS':
                n = int(f.readline())
                self.x = np.empty((n,self.d))

            # Item containing the bounding box.
            elif ' '.join(item[1:3]) == 'BOX BOUNDS':
                d = len(item[3:])
                self.x = np.empty((self.n,d))
                self.box = np.zeros(d, dtype=np.longdouble)

                for c in range(d):
                    boundary = f.readline().split()
                    self.box[c] = float(boundary[1]) - float(boundary[0])

            # Main table contains the per-atom data. Should come at the end.
            elif item[1] == 'ATOMS':
                assert self.n > 0
                assert self.d >= 1 and self.d <= 3
                assert self.box is not None

                headings = item[2:]
                assert 'id' in headings
                assert 'x' or 'xs' in headings

                # Pandas closes the file buffer after reading, so we pass pandas a string buffer
                # with the relevant data in so it will read that without closing the file.
                c = io.StringIO()
                for i in range(n): c.write(f.readline())
                c.seek(0)
                # Read the table from the string buffer.
                table = pandas.read_table(c, sep='\s+', names=headings, nrows=n, iterator=True)
                try: table = table.sort_values('id')
                except: table = table.sort('id')

                if 'xs' in headings:
                    cols = ['xs','ys','zs'][:self.d]
                    self.x = table[cols].values.copy('c').astype(np.longdouble)
                    for c in range(d): self.x[:,c] *= self.box[c]
                else:
                    cols = ['x','y','z'][:self.d]
                    self.x = table[cols].values.copy('c').astype(np.longdouble)

                self.types = np.array(table['type'])
                return

            else: raise RuntimeError('unknown header: %s' % item)

    def write(self, out=sys.stdout):
        """Dump the snapshot to a file in LAMMPS (.atom) format.

        Args:
            out (file or str): file to write the snapshot to
        """
        if type(out) is str: out = open(out, 'w')
        out.write(str(self))
        out.write('\n')

    def __repr__(self):
        """Internal representation of the object for printing to debugger."""
        return '<snapshot n=%r t=%r>' % (self.n, self.time)

    def __str__(self):
        """String representation of the snapshot in LAMMPS (.atom) format"""
        f = io.StringIO()
        f.write('ITEM: TIMESTEP\n%r\n' % self.time)
        f.write('ITEM: NUMBER OF ATOMS\n%r\n' % self.n)
        f.write('ITEM: BOX BOUNDS')
        for _ in self.box: f.write(' pp')
        f.write('\n')
        for length in self.box:
            f.write('0 %.8f\n' % length)
        f.write('ITEM: ATOMS id type x y z')
        for i,(name,x) in enumerate(zip(self.types,self.x)):
            f.write('\n')
            f.write('%r %s' % (i,name))
            for coord in x: f.write(' %.4f' % coord)
        return f.getvalue()

def write(path, trajectory):
    """Write a trajectory to the path.

    Args:
        path (str): location to dump trajectory
        trajectory (list or snapshot): trajectory to dump
    """
    if type(trajectory) is not list: trajectory = [trajectory]

    with open(path, 'w') as f:
        for snap in trajectory:
            snap.write(f)

def read(path, max_frames=None):
    """Read a trajectory from the path.

    >>> list(read('trajectory.atom', 2))
    [<snapshot n=10976 t=0>, <snapshot n=10976 t=1>]

    Args:
        path (str): location of trajectory file
        max_frames (int or None): maximum number of frames to read from the file
    Returns:
        trajectory (generator): generator iterating through the snapshots in the trajectory
    """
    with open(path) as f:
        frames = 0
        while True:
            try: snap = snapshot.from_file(f)
            except NoSnapshotError: break

            yield snap
            frames += 1
            if max_frames is not None and frames is max_frames: break

if __name__ == '__main__':
    # Test to check that the snapshot class is working acceptably.
    n, d = 100, 3
    a = snapshot(np.random.random((n,d)), box = np.ones(d), types = ['A']*n, time=0)
    b = snapshot(np.random.random((n,d)), box = np.ones(d), types = ['A']*n, time=1)
    trajectory = [a,b]
    print('write:', trajectory)

    # Test reading/writing works.
    path = 'test_trajectory.atom'
    write(path, trajectory)
    trajectory = list(read(path))
    print(' read:', trajectory)
