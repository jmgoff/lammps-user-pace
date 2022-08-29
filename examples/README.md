# Notes for descriptor example

## Running Use:

lmp -in in.pace.product

The compute will return the derivatives of the energy with respect to parameters
(expansion coefficients for corresponding ACE descriptors) for all descriptors
defined in the coupling_coefficients.ace file. These values are the first
entries in the A_matrix from fitsnap. Currently, compute pace does not evaluate
forces or virial derivatives.

## Coupling coefficients: coupling_coefficients.ace

This file contains analytical coupling coefficients for the descriptor space
from drautz 2019. These are the generalized Clebsch-Gordan coefficients for the
descriptors defined by lexicographically ordered n,l combinations. Some
analytical couplings (if rank >=4) may be trivial (0-valued). To obtain
non-trivial generalized CG coefficients for all descriptors defined in a .ace
file, use "write_opt_coupling.py" for semi-numerical coupling coefficients.


