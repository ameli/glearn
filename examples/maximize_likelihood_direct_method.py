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
from gaussian_proc.sample_data import generate_points, generate_data
from gaussian_proc.mean import LinearModel
from gaussian_proc.kernels import Matern, Exponential, SquareExponential, \
        RationalQuadratic
from gaussian_proc import Correlation
from gaussian_proc import Covariance
from gaussian_proc import GaussianProcess


# ====
# main
# ====

def main():

    # Generate points
    num_points = 30
    dimension = 2
    grid = True
    points = generate_points(num_points, dimension, grid)

    # Generate noisy data
    noise_magnitude = 0.2
    z = generate_data(points, noise_magnitude)

    # Mean
    mean = LinearModel.design(points, polynomial_degree=2)

    # Correlation
    # kernel = Matern()
    # kernel = Exponential()
    # kernel = SquareExponential()
    kernel = RationalQuadratic()
    cor = Correlation(points, kernel=kernel, distance_scale=0.1, sparse=False)

    # Covariance
    cov = Covariance(cor)

    # Gaussian process
    gp = GaussianProcess(mean, cov)

    # Trainign options
    likelihood_method = 'direct'
    # likelihood_method = 'profiled'

    # optimization_method = 'Nelder-Mead'
    # optimization_method = 'BFGS'         # requires jacobian
    # optimization_method = 'CG'           # requires jacobian
    optimization_method = 'Newton-CG'      # requires jacobian, hessian
    # optimization_method = 'dogleg'       # requires jacobian, hessian
    # optimization_method = 'trust-exact'  # requires jacobian, hessian
    # optimization_method = 'trust-ncg'    # requires jacobian, hessian

    t0 = time.time()
    # gp.train(z, options=options, plot=False)
    gp.train(z, likelihood_method=likelihood_method,
             optimization_method=optimization_method)
    t1 = time.time()
    print('Elapsed time: %0.2f' % (t1 - t0))


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(main())
