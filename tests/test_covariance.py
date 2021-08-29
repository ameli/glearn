#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import sys
import numpy
import numpy.linalg
from scipy.sparse import isspmatrix
from gaussian_proc.sample_data import generate_points
from gaussian_proc.kernels import Matern
from gaussian_proc import Correlation, Covariance


# ==============
# relative error
# ==============

def relative_error(estimate, exact):
    """
    Compute the relative error of an estimate, in percent.
    """

    tol = 1e-15
    if numpy.linalg.norm(exact) < tol:
        if numpy.linalg.norm(estimate - exact) < tol:
            relative_error = 0.0
        else:
            relative_error = numpy.inf
    else:
        relative_error = numpy.linalg.norm(estimate - exact) / \
                numpy.linalg.norm(exact) * 100.0

    return relative_error


# ===============
# check functions
# ===============

def check_functions(
        function,
        K,
        I,                                                         # noqa: E741
        cov,
        x,
        sigmas,
        sigma0s,
        exponents,
        error_rtol):
    """
    Checks ``gaussian_proc.Covariance`` function.
    """

    success = True

    for sigma in sigmas:
        for sigma0 in sigma0s:
            for exponent in exponents:

                # Matrix
                S = I
                S1 = sigma**2 * K + sigma0**2 * I
                for i in range(1, exponent+1):
                    S = S1 @ S

                # check trace
                if function == 'trace':
                    y0 = numpy.trace(S)
                    y1 = cov.trace(sigma, sigma0, exponent=exponent)

                # check traceinv
                elif function == 'traceinv':
                    if sigma != 0 or sigma0 != 0:
                        y0 = numpy.trace(numpy.linalg.inv(S))
                        y1 = cov.traceinv(sigma, sigma0, exponent=exponent)
                    else:
                        y0 = None

                # check logdet
                elif function == 'logdet':
                    if sigma != 0 or sigma0 != 0:
                        y0 = numpy.log(numpy.linalg.det(S))
                        y1 = cov.logdet(sigma, sigma0, exponent=exponent)
                    else:
                        y0 = None

                # check solve
                elif function == 'solve':
                    if sigma != 0 or sigma0 != 0:
                        y0 = numpy.linalg.solve(S, x)
                        y1 = cov.solve(sigma, sigma0, x, exponent)
                    else:
                        y0 = None

                # check dot
                elif function == 'dot':
                    y0 = S @ x
                    y1 = cov.dot(sigma, sigma0, x, exponent=exponent)

                # Check error
                if y0 is not None:
                    error = relative_error(y1, y0)
                    if error > error_rtol:
                        if success:
                            print('\tcheck dot:      Failed')
                            success = False
                        print('\t\tsigma: %0.2f, sigma0: %0.2f, exponent: %d:'
                              % (sigma, sigma0, exponent), end='')
                        print('\tdot error: %0.4f%%'
                              % error)

    if success:
        print('\tcheck dot:      OK')


# ===============
# test covariance
# ===============

def test_covariance():
    """
    A test for :mod:`gaussian_proc.Covariance` sub-package.
    """

    # Parameters combinations to be tested
    sigmas = [0.0, 2.3]
    sigma0s = [0.0, 3.2]
    exponents = [0, 1, 2, 3]
    sparses = [False, True]
    imate_methods = ['eigenvalue', 'cholesky', 'hutchinson', 'slq']
    functions = ['trace', 'traceinv', 'logdet', 'solve', 'dot']

    # Generate points
    num_points = 20
    dimension = 2
    grid = True
    points = generate_points(num_points, dimension, grid)

    # Correlation
    kernel = Matern()

    for sparse in sparses:

        print('--------------------------')
        print('Using sparse matrix: %s' % sparse)
        print('--------------------------\n')

        cor = Correlation(points, kernel=kernel, distance_scale=0.1,
                          sparse=sparse, density=0.01)

        K = cor.get_matrix()
        I = numpy.eye(K.shape[0])                                  # noqa: E741

        # A column vector
        x = numpy.arange(K.shape[0]).astype(float)

        # Check each function
        for imate_method in imate_methods:

            # eigenvalue method cannot process sparse matrices.
            if imate_method == 'eigenvalue' and isspmatrix(K):
                continue

            # For stochastic methods, use a less restrictive error tolerance
            if imate_method in ['hutchinson', 'slq']:
                error_rtol = 5  # in percent
            else:
                error_rtol = 1e-8  # in percent

            # Covariance
            cov = Covariance(cor, imate_method=imate_method)
            print('imate method: %s' % imate_method)

            for function in functions:
                check_functions(function, K, I, cov, x, sigmas, sigma0s,
                                exponents, error_rtol)

            print('')


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_covariance())
