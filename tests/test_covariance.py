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
import scipy
from glearn import sample_data
from glearn.kernels import Matern
from glearn import Covariance

import warnings
warnings.resetwarnings()
warnings.filterwarnings("error")


# ==============
# relative error
# ==============

def relative_error(estimate, exact, tol=1e-13):
    """
    Compute the relative error of an estimate, in percent.
    """

    if numpy.isscalar(estimate):
        estimate = numpy.array([estimate])
    else:
        # Ravel matrix data
        estimate = estimate.ravel()

    if numpy.isscalar(exact):
        exact = numpy.array([exact])
    else:
        # Ravel matrix data
        exact = exact.ravel()

    relative_error = numpy.zeros_like(estimate)

    for i in range(relative_error.size):

        if (numpy.isinf(estimate[i]) and numpy.isinf(exact[i])) or \
                (numpy.isinf(-estimate[i]) and numpy.isinf(-exact[i])):
            relative_error[i] = 0.0
        else:

            # Ravel matrix data
            difference = numpy.abs(estimate[i] - exact[i])

            if numpy.linalg.norm(exact[i]) < tol:
                if numpy.linalg.norm(difference) < tol:
                    relative_error[i] = 0.0
                else:
                    relative_error[i] = numpy.inf
            else:
                relative_error[i] = numpy.linalg.norm(difference) / \
                        numpy.linalg.norm(exact[i]) * 100.0

    return numpy.max(relative_error)


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
            for p in exponents:
                for derivative in derivatives:

                    # If derivative is more than zero, the matrix is no longer
                    # positive definite. So, test only those functions that
                    # does not need positive-definiteness.
                    if len(derivative) > 0:
                        if function in ('solve', 'traceinv'):
                            continue
                        if cov.mixed_cor.imate_options['method'] == 'cholesky':
                            continue
                        if cov.mixed_cor.imate_options['method'] == \
                                'hutchinson' and \
                                function == 'logdet':
                            continue

                    # This case is not implemented
                    if p > 1 and len(derivative) > 0:
                        continue

                    K = cov.cor.get_matrix(derivative=derivative)
                    n = K.shape[0]
                    if scipy.sparse.issparse(K):
                        I = scipy.sparse.eye(n, format='csr')      # noqa: E741
                    else:
                        I = numpy.eye(n)                           # noqa: E741

                    # A column vector
                    m = 3
                    X = numpy.arange(m * n).astype(float)
                    X = X.reshape((n, m))

                    if len(derivative) == 0:
                        S1 = sigma**2 * K + sigma0**2 * I
                    else:
                        # Here, the input K is the derivative of some matrix
                        S1 = sigma**2 * K

                    if p == 0 and len(derivative) > 0:
                        if scipy.sparse.issparse(K):
                            S = scipy.sparse.csr_array((n, n), dtype=float)
                        else:
                            S = numpy.zeros_like(I)
                    else:
                        S = I

                    # Perform exponentiation to form the matrix
                    for i in range(1, p+1):
                        S = S1 @ S

                    # check trace
                    if function == 'trace':
                        if scipy.sparse.issparse(S):
                            y0 = S.trace()
                        else:
                            y0 = numpy.trace(S)
                        y1 = cov.trace(sigma, sigma0, p=p,
                                       derivative=derivative)

                    # check traceinv
                    elif function == 'traceinv':
                        if sigma != 0 or sigma0 != 0:
                            if p == 0 and len(derivative) > 0:
                                y0 = numpy.nan
                            elif sigma == 0 and len(derivative) > 0:
                                y0 = numpy.nan
                            else:
                                if scipy.sparse.issparse(S):
                                    Sinv = scipy.sparse.linalg.inv(S.tocsc())
                                    y0 = Sinv.trace()
                                else:
                                    y0 = numpy.trace(numpy.linalg.inv(S))
                            y1 = cov.traceinv(sigma, sigma0, p=p,
                                              derivative=derivative)
                        else:
                            y0 = None

                    # check logdet
                    elif function == 'logdet':
                        if sigma != 0 or sigma0 != 0:
                            with numpy.errstate(over='raise'):
                                try:
                                    if scipy.sparse.issparse(S):
                                        det_S = numpy.linalg.det(S.toarray())
                                    else:
                                        det_S = numpy.linalg.det(S)
                                    if numpy.abs(det_S) < 1e-15:
                                        y0 = -numpy.inf
                                    else:
                                        y0 = numpy.log(det_S)
                                except FloatingPointError:
                                    if scipy.sparse.issparse(S):
                                        eig_S = numpy.linalg.eig(
                                            S.toarray())[0]
                                    else:
                                        eig_S = numpy.linalg.eig(S)[0]
                                    y0 = numpy.sum(numpy.log(
                                        eig_S.astype(numpy.complex128)))
                                    imag = numpy.mod(y0.imag, numpy.pi)
                                    imag = numpy.abs(imag)
                                    tol = 1e-7
                                    if (imag < tol) or \
                                       (numpy.abs((imag - numpy.pi)) < tol):
                                        y0 = y0.real
                                    else:
                                        raise RuntimeError(
                                            'logdet of S is a complex number: '
                                            'real: %e, imag: %e'
                                            % (y0.real, imag))
                            y1 = cov.logdet(sigma, sigma0, p=p,
                                            derivative=derivative)
                        else:
                            y0 = None

                    # check solve
                    elif function == 'solve':
                        if sigma != 0 or sigma0 != 0:
                            if p == 0 and len(derivative) > 0:
                                y0 = numpy.zeros_like(X)
                                y0[:] = numpy.nan
                            elif sigma == 0 and len(derivative) > 0:
                                y0 = numpy.zeros_like(X)
                                y0[:] = numpy.nan
                            else:
                                if scipy.sparse.issparse(S):
                                    y0 = scipy.sparse.linalg.spsolve(S, X)
                                else:
                                    y0 = numpy.linalg.solve(S, X)
                            y1 = cov.solve(X, sigma=sigma, sigma0=sigma0, p=p,
                                           derivative=derivative)
                        else:
                            y0 = None

                    # check dot
                    elif function == 'dot':
                        y0 = S @ X
                        y1 = cov.dot(X, sigma=sigma, sigma0=sigma0, p=p,
                                     derivative=derivative)

                    # Check error
                    if y0 is not None:
                        error = relative_error(y1, y0)
                        if numpy.any(error > error_rtol):
                            if success:
                                print('\tcheck %8s: Failed' % function)
                                success = False
                            print('\t\tsigma: %0.2f, ' % sigma, end="")
                            print('sigma0: %0.2f, ' % sigma0, end="")
                            print('exponent: %d, ' % p, end="")
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
    points = sample_data.generate_points(num_points, dimension=dimension,
                                         grid=grid)

    # Correlation
    kernel = Matern()

    for sparse in sparses:

        print('--------------------------')
        print('Using sparse matrix: %s' % sparse)
        print('--------------------------\n')

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

            imate_options = {'method': imate_method}

            # Covariance
            cov = Covariance(points, kernel=kernel, scale=0.1, sparse=sparse,
                             density=0.01, imate_options=imate_options)
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
