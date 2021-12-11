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
from glearn.sample_data import generate_points
from glearn.kernels import Matern
from glearn import Correlation, Covariance


# ==============
# relative error
# ==============

def relative_error(estimate, exact, tol=1e-13):
    """
    Compute the relative error of an estimate, in percent.
    """

    # Ravel matrix data
    difference = estimate - exact
    if not numpy.isscalar(difference):
        difference = difference.ravel()
        exact = exact.ravel()

    if numpy.linalg.norm(exact) < tol:
        if numpy.linalg.norm(difference) < tol:
            relative_error = 0.0
        else:
            relative_error = numpy.inf
    else:
        relative_error = numpy.linalg.norm(difference) / \
                numpy.linalg.norm(exact) * 100.0

    return relative_error


# ===============
# check functions
# ===============

def check_functions(
        function,
        cov,
        sigmas,
        sigma0s,
        exponents,
        derivatives,
        error_rtol):
    """
    Checks ``glearn.Covariance`` function.
    """

    success = True

    for sigma in sigmas:
        for sigma0 in sigma0s:
            for exponent in exponents:
                for derivative in derivatives:

                    # If derivative is more than zero, the matrix is no longer
                    # positive definite. So, test only those functions that
                    # does not need positive-definiteness.
                    if len(derivative) > 0:
                        if function in ('solve', 'traceinv'):
                            continue
                        if cov.mixed_cor.imate_method == 'cholesky':
                            continue
                        if cov.mixed_cor.imate_method == 'hutchinson' and \
                                function == 'logdet':
                            continue

                    # This case is not implemented
                    if exponent > 1 and len(derivative) > 0:
                        continue

                    K = cov.cor.get_matrix(derivative=derivative)
                    I = numpy.eye(K.shape[0])                      # noqa: E741

                    # A column vector
                    m = 3
                    X = numpy.arange(m * K.shape[0]).astype(float)
                    X = X.reshape((K.shape[0], m))

                    if len(derivative) == 0:
                        S1 = sigma**2 * K + sigma0**2 * I
                    else:
                        # Here, the input K is the derivative of some matrix
                        S1 = sigma**2 * K

                    if exponent == 0 and len(derivative) > 0:
                        S = numpy.zeros_like(I)
                    else:
                        S = I

                    # Perform exponentiation to form the matrix
                    for i in range(1, exponent+1):
                        S = S1 @ S

                    # check trace
                    if function == 'trace':
                        y0 = numpy.trace(S)
                        y1 = cov.trace(sigma, sigma0, exponent=exponent,
                                       derivative=derivative)

                    # check traceinv
                    elif function == 'traceinv':
                        if sigma != 0 or sigma0 != 0:
                            if exponent == 0 and len(derivative) > 0:
                                y0 = numpy.nan
                            elif sigma == 0 and len(derivative) > 0:
                                y0 = numpy.nan
                            else:
                                y0 = numpy.trace(numpy.linalg.inv(S))
                            y1 = cov.traceinv(sigma, sigma0,
                                              exponent=exponent,
                                              derivative=derivative)
                        else:
                            y0 = None

                    # check logdet
                    elif function == 'logdet':
                        if sigma != 0 or sigma0 != 0:
                            y0 = numpy.log(numpy.linalg.det(S))
                            y1 = cov.logdet(sigma, sigma0, exponent=exponent,
                                            derivative=derivative)
                        else:
                            y0 = None

                    # check solve
                    elif function == 'solve':
                        if sigma != 0 or sigma0 != 0:
                            if exponent == 0 and len(derivative) > 0:
                                y0 = numpy.zeros_like(X)
                                y0[:] = numpy.nan
                            elif sigma == 0 and len(derivative) > 0:
                                y0 = numpy.zeros_like(X)
                                y0[:] = numpy.nan
                            else:
                                y0 = numpy.linalg.solve(S, X)
                            y1 = cov.solve(X, sigma=sigma, sigma0=sigma0,
                                           exponent=exponent,
                                           derivative=derivative)
                        else:
                            y0 = None

                    # check dot
                    elif function == 'dot':
                        y0 = S @ X
                        y1 = cov.dot(X, sigma=sigma, sigma0=sigma0,
                                     exponent=exponent, derivative=derivative)

                    # Check error
                    if y0 is not None:
                        error = relative_error(y1, y0)
                        if error > error_rtol:
                            if success:
                                print('\tcheck %8s: Failed' % function)
                                success = False
                            print('\t\tsigma: %0.2f, ' % sigma, end="")
                            print('sigma0: %0.2f, ' % sigma0, end="")
                            print('exponent: %d, ' % exponent, end="")
                            print('derivative: %s,' % derivative, end="")
                            print('\terror: %0.4f%%' % error)

    if success:
        print('\tcheck %8s: OK' % function)


# ===============
# test covariance
# ===============

def test_covariance():
    """
    A test for :mod:`glearn.Covariance` sub-package.
    """

    # Parameters combinations to be tested
    sigmas = [0.0, 2.3]
    sigma0s = [0.0, 3.2]
    exponents = [0, 1, 2, 3]
    derivatives = [[], [0], [1], [0, 0], [0, 1], [1, 0], [1, 1]]
    sparses = [False, True]
    imate_methods = ['eigenvalue', 'cholesky', 'hutchinson', 'slq']
    functions = ['trace', 'traceinv', 'logdet', 'solve', 'dot']

    # Generate points
    num_points = 20
    dimension = 2
    grid = True
    points = generate_points(num_points, dimension=dimension, grid=grid)

    # Correlation
    kernel = Matern()

    for sparse in sparses:

        print('--------------------------')
        print('Using sparse matrix: %s' % sparse)
        print('--------------------------\n')

        cor = Correlation(points, kernel=kernel, scale=0.1, sparse=sparse,
                          density=0.01)

        # Check each function
        for imate_method in imate_methods:

            # eigenvalue method cannot process sparse matrices.
            if imate_method == 'eigenvalue' and sparse:
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
                check_functions(function, cov, sigmas, sigma0s, exponents,
                                derivatives, error_rtol)

            print('')


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_covariance())
