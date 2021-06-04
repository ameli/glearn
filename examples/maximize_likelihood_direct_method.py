#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import sys
import time
from _utilities.data_utilities import generate_points, generate_data, \
        generate_basis_functions
from gaussian_proc import generate_correlation
from gaussian_proc import GaussianProcess


# ====
# main
# ====

def main():

    # Generate points
    num_points = 50
    dimension = 2
    grid = True
    points = generate_points(num_points, dimension, grid)

    # Generate noisy data
    noise_magnitude = 0.2
    z = generate_data(points, noise_magnitude)

    # Generate Correlation Matrix
    correlation_scale = 0.1
    nu = 0.5
    density = 1e-2
    sparse = False
    K = generate_correlation(points, correlation_scale, nu, grid,
                             sparse=sparse, density=density)

    # generate basis functions
    X = generate_basis_functions(points, polynomial_degree=2,
                                 trigonometric=False)

    # Gaussian process
    likelihood_method = 'direct'
    # likelihood_method = 'profiled'
    gaussian_process = GaussianProcess(
            X, K, likelihood_method=likelihood_method)

    t0 = time.time()
    gaussian_process.train(z, plot=False)
    t1 = time.time()
    print('Elapsed time: %0.2f' % (t1 - t0))


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(main())
