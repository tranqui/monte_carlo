# Hard Sphere Monte Carlo

## Introduction

This code forms the basis of a short course on simulation techniques for PhD candidates at the Bristol Centre for Functional Nanomaterials (BCFN) centre for doctoral training. The course teaches the principles of simulation techniques and coding.

This repository contains python code which emphasises **clarity** over speed, to provide the simplest introduction for students encountering the concepts for the first time. As such, the techniques presented should not be considered state-of-the-art. In fact, the code will run **very** slow for systems with large numbers of particles, or at high densities.

The course was taught with this repository by C. Patrick Royall the first time in Autumn 2016 at the University of Bristol.

## Overview of the files

The core files included are:
* main.py: the main script that brings everything together: it starts, runs and analyses the simulation. The student will be expected to read, understand and adjust the parameters in this script without adding additional code.
* montecarlo.py: the module which runs the simulation itself; this is the main file students are expected to write new code for. In fact, when given to the students this will initially have none of the functions defined, instead they will be expected to write the body of the functions (after some introduction within the lectures).

Additional helper modules are provided which hide some of the more complex tasks from the student that are not essential for understanding the simulation.  The student will not be expected to read/change these, unless they are especially interested. These are as follows:
* g.py: module which calculates radial distribution functions g(r) from a trajectory.
* atom.py: module which defines a 'snapshot' class which stores a configuration, and reads/writes simulation trajectories to the disk
* lattice.py: module for generating crystal lattices for the initial condition of the simulation

## Authors

* Joshua F. Robinson <joshua.robinson@bristol.ac.uk>
* Peter Crowther <pc9836@bristol.ac.uk>

Based on code by C. Patrick Royall <chcpr@bristol.ac.uk>

Group website: [padrus.com](http://padrus.com)