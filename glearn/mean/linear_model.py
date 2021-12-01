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

import numpy


# ============
# linear model
# ============

class LinearModel(object):
    """
    """

    # ====
    # init
    # ====

    def __init__(
            self,
            points,
            polynomial_degree=0,
            trigonometric_coeff=None,
            hyperbolic_coeff=None,
            fun=None,
            b=None,
            B=None):
        """
        """

        trigonometric_coeff, hyperbolic_coeff = self._check_arguments(
                points, polynomial_degree, trigonometric_coeff,
                hyperbolic_coeff, fun)

        # If points are 1d array, wrap them to a 2d array
        if points.ndim == 1:
            points = numpy.array([points], dtype=float).T

        # Store function info
        self.points = points
        self.polynomial_degree = polynomial_degree
        self.trigonometric_coeff = trigonometric_coeff
        self.hyperbolic_coeff = hyperbolic_coeff
        self.fun = fun

        # Generate design matrix
        self.X = self.generate_design_matrix(self.points)

        # Check b and B for their size consistency with X
        b, B = self._check_b_B(b, B)

        self.b = b        # Prior mean of beta
        self.B = B        # Prior covariance of beta
        self.beta = None  # Posterior mean of beta (will be computed)
        self.C = None     # Posterior covariance of beta (will be computed)

        # Precision of the prior of beta
        if self.B is not None:
            self.Binv = numpy.linalg.inv(self.B)
        else:
            # When B is None, we assume it is infinity. Hence, the precision
            # matrix (inverse of covariance) will be zero matrix.
            m = self.X.shape[1]
            self.Binv = numpy.zeros((m, m), dtype=float)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(
            self,
            points,
            polynomial_degree,
            trigonometric_coeff,
            hyperbolic_coeff,
            fun):
        """
        """

        # Check points
        if points is None:
            raise ValueError('"points" cannot be "None".')

        elif not isinstance(points, numpy.ndarray):
            raise TypeError('"points" should be a "numpy.ndarray" type.')

        # Check at least one of polynomial, trigonometric, hyperbolic, or fun
        # is given.
        if (polynomial_degree is None) and (trigonometric_coeff is None) and \
           (hyperbolic_coeff is None) and (fun is None):
            raise ValueError('At least, one of "polynomial_degree", ' +
                             '"trigonometric_coeff", "hyperbolic_coeff", ' +
                             'or "fun" must be set.')

        # Check polynomial degree
        if polynomial_degree is not None:

            if not isinstance(polynomial_degree, int):
                raise ValueError('"polynomial_degree" must be an integer.')

            elif polynomial_degree < 0:
                raise ValueError('"polynomial_degree" should be non-negative.')

        # Check trigonometric coeff
        if trigonometric_coeff is not None:

            if numpy.isscalar(trigonometric_coeff):
                if not isinstance(trigonometric_coeff, int) and \
                   not isinstance(trigonometric_coeff, float):
                    raise ValueError('"trigonometric_coeff" must be a float ' +
                                     'type.')

                # Convert scalar to numpy array
                trigonometric_coeff = numpy.array([trigonometric_coeff],
                                                  dtype=float)

            elif isinstance(trigonometric_coeff, list):
                # Convert list to numpy array
                trigonometric_coeff = numpy.array(trigonometric_coeff,
                                                  dtype=float)
            elif not isinstance(trigonometric_coeff, numpy.ndarray):
                raise TypeError('"trigonometric_coeff" should be a scalar, ' +
                                ', a list, or an array.')
            elif trigonometric_coeff.ndim > 1:
                raise ValueError('"trigonometric_coeff" should be a 1d array.')

        # Check polynomial degree
        if hyperbolic_coeff is not None:

            if numpy.isscalar(hyperbolic_coeff):
                if not isinstance(hyperbolic_coeff, int) and \
                   not isinstance(hyperbolic_coeff, float):
                    raise ValueError('"hyperbolic_coeff" must be a float ' +
                                     'type.')

                # Convert scalar to numpy array
                hyperbolic_coeff = numpy.array([hyperbolic_coeff], dtype=float)

            elif isinstance(hyperbolic_coeff, list):
                # Convert list to numpy array
                hyperbolic_coeff = numpy.array(hyperbolic_coeff, dtype=float)
            elif not isinstance(hyperbolic_coeff, numpy.ndarray):
                raise TypeError('"hyperbolic_coeff" should be a scalar, ' +
                                ', a list, or an array.')
            elif hyperbolic_coeff.ndim > 1:
                raise ValueError('"hyperbolic_coeff" should be a 1d array.')

        # Check fun
        if fun is not None and not callable(fun):
            raise TypeError('"fun" should be a callable function or None.')

        return trigonometric_coeff, hyperbolic_coeff

    # =========
    # check b B
    # =========

    def _check_b_B(self, b, B):
        """
        Checks the sizes of and B to be consistent with self.X.
        """

        # Check b
        if b is not None:
            if numpy.isscalar(b):

                if self.X.ndim != 1 or self.X.shape[1] != 1:
                    raise ValueError('"b" should be a vector of the same ' +
                                     'size as the number of columns of the ' +
                                     'design matrix "X".')
                else:
                    # Convert scalar to a 1d vector of unit size
                    b = numpy.array([b], dtype=float)

            elif b.size != self.X.shape[1]:
                raise ValueError('"b" should have the same size as the ' +
                                 'number of columns of the design matrix ' +
                                 '"X", which is %d.' % self.X.shape[1])

            if not isinstance(b, numpy.ndarray):
                raise TypeError('"b" should be either a scalar (if the' +
                                'design matrix is a column vector) or a row' +
                                'vector of "numpy.ndarray" type.')

        # Check B
        if B is not None:

            if b is None:
                raise ValueError('When "B" is given, "b" cannot be None.')
            elif numpy.isscalar(b) and not numpy.isscalar(B):
                raise ValueError('When "b" is a scalar, "B" should also be a' +
                                 'scalar.')
            elif not isinstance(B, numpy.ndarray):
                raise TypeError('"B" should be a "numpy.ndarray" type.')

            elif numpy.isscalar(B):
                if self.X.ndim != 1 or self.X.shape[1] != 1:
                    raise ValueError('"b" should be a vector of the same ' +
                                     'size as the number of columns of the ' +
                                     'design matrix "X".')
                else:
                    # Convert scalar to a 2d vector of unit size
                    B = numpy.array([[B]], dtype=float)

            elif B.shape != (b.size, b.size):
                raise ValueError('"B" should be a square matrix with the' +
                                 'same number of columns/rows as the size of' +
                                 'vector "b".')

        return b, B

    # ======================
    # generate design matrix
    # ======================

    def generate_design_matrix(self, points):
        """
        Generates design matrix (basis functions) for the mean function of the
        linear model.
        """

        # Convert a vector to matrix if dimension is one
        if points.ndim == 1:
            points = numpy.array([points]).T

        # Initialize output
        X_list = []

        # Polynomial basis functions
        if self.polynomial_degree is not None:
            X_polynomial = self._generate_polynomial_basis(points)
            X_list.append(X_polynomial)

        # Trigonometric basis functions
        if self.trigonometric_coeff is not None:
            X_trigonometric = self._generate_trigonometric_basis(points)
            X_list.append(X_trigonometric)

        # Hyperbolic basis functions
        if self.hyperbolic_coeff is not None:
            X_hyperbolic = self._generate_hyperbolic_basis(points)
            X_list.append(X_hyperbolic)

        # Custom function basis functions
        if self.fun is not None:
            X_fun = self._generate_custom_basis(points)
            X_list.append(X_fun)

        # Concatenate those bases that are not None
        if len(X_list) == 0:
            raise RuntimeError('No basis was generated.')
        elif len(X_list) == 1:
            X = X_list[0]
        else:
            X = numpy.concatenate(X_list, axis=1)

        return X

    # =========================
    # generate polynomial basis
    # =========================

    def _generate_polynomial_basis(self, points):
        """
        Generates polynomial basis functions.
        """

        n = points.shape[0]
        dimension = points.shape[1]

        # Adding polynomial functions
        powers_array = numpy.arange(self.polynomial_degree + 1)
        powers_tile = numpy.tile(powers_array, (dimension, 1))
        powers_mesh = numpy.meshgrid(*powers_tile)

        powers_ravel = []
        for i in range(dimension):
            powers_ravel.append(powers_mesh[i].ravel())

        # The array powers_ravel contains all combinations of powers
        powers_ravel = numpy.array(powers_ravel)

        # For each combination of powers, we compute the power sum
        powers_sum = numpy.sum(powers_ravel, axis=0)

        # The array powers contains those combinations that their sum does
        # not exceed the polynomial_degree
        powers = powers_ravel[:, powers_sum <= self.polynomial_degree]

        num_degrees = powers.shape[0]
        num_basis = powers.shape[1]

        # Basis functions
        X_polynomial = numpy.ones((n, num_basis), dtype=float)
        for j in range(num_basis):
            for i in range(num_degrees):
                X_polynomial[:, j] *= points[:, i]**powers[i, j]

        return X_polynomial

    # ============================
    # generate trigonometric basis
    # ============================

    def _generate_trigonometric_basis(self, points):
        """
        Generates trigonometric basis functions.
        """

        n = points.shape[0]
        dimension = points.shape[1]

        tri_size = self.trigonometric_coeff.size
        X_trigonometric = numpy.empty((n, 2*dimension*tri_size))

        for i in range(tri_size):
            for j in range(dimension):
                X_trigonometric[:, 2*dimension*i + 2*j+0] = numpy.sin(
                        points[:, j] * self.trigonometric_coeff[i])
                X_trigonometric[:, 2*dimension*i + 2*j+1] = numpy.cos(
                        points[:, j] * self.trigonometric_coeff[i])

        return X_trigonometric

    # =========================
    # generate hyperbolic basis
    # =========================

    def _generate_hyperbolic_basis(self, points):
        """
        Generate hyperbolic basis functions.
        """

        n = points.shape[0]
        dimension = points.shape[1]

        hyp_size = self.hyperbolic_coeff.size
        X_hyperbolic = numpy.empty((n, 2*dimension*hyp_size))

        for i in range(hyp_size):
            for j in range(dimension):
                X_hyperbolic[:, 2*dimension*i + 2*j+0] = numpy.sinh(
                        points[:, j] * self.hyperbolic_coeff[i])
                X_hyperbolic[:, 2*dimension*i + 2*j+1] = numpy.cosh(
                        points[:, j] * self.hyperbolic_coeff[i])

        return X_hyperbolic

    # =====================
    # generate custom basis
    # =====================

    def _generate_custom_basis(self, points):
        """
        Generate custom basis functions.
        """

        n = points.shape[0]

        # Generate on the first point to see what is the size of the output
        X_fun_init = self.fun(points[0, :])

        if X_fun_init.ndim != 1:
            raise ValueError('"fun" should output 1d array.')

        # Initialize output 2D array for all points
        X_fun = numpy.empty((n, X_fun_init.size), dtype=float)
        X_fun[0, :] = X_fun_init

        for i in range(1, n):
            X_fun[i, :] = self.fun(points[i, :])

        return X_fun

    # =================
    # update hyperparam
    # =================

    def update_hyperparam(self, cov, z):
        """
        Updates (or computes, if has not been done so) the mean and covariance
        of the posterior of the parameter beta. This function should be called
        after training, and after cov itself is updated from the training, and
        before prediction.
        """

        # Note: cov should has been updated already after training.
        sigma, sigma0 = cov.get_sigmas()

        # Posterior covariance of beta
        Y = cov.solve(self.X, sigma=sigma, sigma0=sigma0)
        Cinv = numpy.matmul(self.X.T, Y)

        # Note: B in this class is B1 in the paper notations. That is, self.B
        # here means B1 = B / (sigma**2).
        if self.B is not None:
            Cinv += self.Binv / (sigma**2)

        self.C = numpy.linalg.inv(Cinv)

        # Posterior mean of beta
        v = numpy.dot(Y.T, z)
        if self.B is not None:
            Binvb = numpy.dot(self.Binv, self.b) / (sigma**2)
            v += Binvb
        self.beta = numpy.matmul(self.C, v)
