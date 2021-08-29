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

    def __init__(self, X, beta_mean=None, beta_cov=None):
        """
        """

        self.check_arguments(X, beta_mean, beta_cov)

        self.X = X
        self.beta_mean = beta_mean
        self.beta_cov = beta_cov

    # ======
    # design
    # ======

    @classmethod
    def design(
            cls,
            points,
            polynomial_degree=0,
            trigonometric_coeff=None,
            hyperbolic_coeff=None,
            beta_mean=None,
            beta_cov=None):
        """
        An alternative costructor used when the desig matrix ``X`` is not
        know. This method designs the design matrix ``X`` and returns a class
        object with the computed ``X``.
        """

        # Generate design matrix X
        X = LinearModel.generate_design_matrix(points, polynomial_degree,
                                               trigonometric_coeff,
                                               hyperbolic_coeff)

        return cls(X, beta_mean, beta_cov)

    # ===============
    # check arguments
    # ===============

    def check_arguments(self, X, beta_mean, beta_cov):
        """
        """

        # Check X
        if X is None:
            raise ValueError('Design matrix "X" cannot be "None".')

        elif not isinstance(X, numpy.ndarray):
            raise TypeError('Design matrix "X" must be of type ' +
                            '"numpy.ndarray".')
        elif X.ndim != 1 and X.ndim != 2:
            raise ValueError('Design matrix "X" should be a 1D column ' +
                             'vector or 2D matrix.')
        elif X.shape[0] < X.shape[1]:
            raise ValueError('Design matrix "X" should have full column rank.')

        # Check beta_mean
        if beta_mean is not None:

            if numpy.isscalar(beta_mean) and (X.ndim != 1 or X.shape[1] != 1):
                raise ValueError('"beta_mean" should be a vector of the ' +
                                 'same size as the number of columns of the ' +
                                 'design matrix "X".')
            elif not isinstance(beta_mean, numpy.ndarray):
                raise TypeError('"beta_mean" should be either a scalar (if ' +
                                'the design matrix is a column vector) ' +
                                'or a row vector of "numpy.ndarray" type.')
            elif beta_mean.size != X.shape[1]:
                raise ValueError('"beta_mean" should have the same size as ' +
                                 'the number of columns of the design ' +
                                 'matrix "X".')

        # Check beta_cov
        if beta_cov is not None:

            if numpy.isscalar(beta_mean) and not numpy.isscalar(beta_cov):
                raise ValueError('When "beta_mean" is a scalar, "beta_cov" ' +
                                 'should also be a scalar.')
            elif not isinstance(beta_cov, numpy.ndarray):
                raise TypeError('"beta_cov" should be a "numpy.ndarray" type.')

            elif beta_cov.shape != (beta_mean.size, beta_mean.size):
                raise ValueError('"beta_cov" should be a square matrix with ' +
                                 'the same number of columns/rows as the ' +
                                 'size of vector "beta_mean".')

    # ======================
    # generate design matrix
    # ======================

    @staticmethod
    def generate_design_matrix(
            points,
            polynomial_degree=0,
            trigonometric_coeff=None,
            hyperbolic_coeff=None):
        """
        Generates design matrix (basis functions) for the mean function of the
        linear model.
        """

        # Check points
        if points is None:
            raise ValueError('"points" cannot be "None".')

        elif not isinstance(points, numpy.ndarray):
            raise TypeError('"points" should be a "numpy.ndarray" type.')

        # Check other arguments
        if (polynomial_degree is None) and (trigonometric_coeff is None) and \
           (hyperbolic_coeff is None):
            raise ValueError('At least, one of "polynomial_degree", ' +
                             '"trigonometric_coeff", or ' +
                             '"hyperbolic_coeff" must be set.')

        # Check polynomial degree
        if polynomial_degree is not None:

            if not isinstance(polynomial_degree, int):
                raise ValueError('"polynomial_degree" must be an integer.')

            elif polynomial_degree < 0:
                raise ValueError('"polynomial_degree" should be non-negative.')

        # Check trigonometric coeff
        if trigonometric_coeff is not None:

            if not isinstance(trigonometric_coeff, int) and \
               not isinstance(trigonometric_coeff, float):
                raise ValueError('"trigonometric_coeff" must be a float type.')

        # Check polynomial degree
        if hyperbolic_coeff is not None:
            if not isinstance(hyperbolic_coeff, int) and \
               not isinstance(hyperbolic_coeff, float):
                raise ValueError('"hyperbolic_coeff" must be a float type.')

        # Convert a vector to matrix if dimension is one
        if points.ndim == 1:
            points = numpy.array([points]).T

        n = points.shape[0]
        dimension = points.shape[1]

        # Initialize output
        X_list = []

        # Polynomial basis functions
        if polynomial_degree is not None:

            # Adding polynomial functions
            powers_array = numpy.arange(polynomial_degree + 1)
            # powers_tile = numpy.tile(powers_array, (polynomial_degree+1, 1))
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
            powers = powers_ravel[:, powers_sum <= polynomial_degree]

            num_degrees = powers.shape[0]
            num_basis = powers.shape[1]

            # Basis functions
            X_polynomial = numpy.ones((n, num_basis), dtype=float)
            for j in range(num_basis):
                for i in range(num_degrees):
                    X_polynomial[:, j] *= points[:, i]**powers[i, j]

            # append to the output array
            X_list.append(X_polynomial)

        # Trigonometric basis functions
        if trigonometric_coeff is not None:
            X_trigonometric = numpy.empty((n, 2*dimension))

            for i in range(dimension):
                X_trigonometric[:, 2*i+0] = numpy.sin(
                        points[:, i] * trigonometric_coeff)
                X_trigonometric[:, 2*i+1] = numpy.cos(
                        points[:, i] * trigonometric_coeff)

            # append to the output list
            X_list.append(X_trigonometric)

        # Hyperbolic basis functions
        if hyperbolic_coeff is not None:
            X_hyperbolic = numpy.empty((n, 2*dimension))

            for i in range(dimension):
                X_hyperbolic[:, 2*i+0] = numpy.sinh(
                        points[:, i] * hyperbolic_coeff)
                X_hyperbolic[:, 2*i+1] = numpy.cosh(
                        points[:, i] * hyperbolic_coeff)

            # append to the output list
            X_list.append(X_hyperbolic)

        # Concatenate those bases that are not None
        if len(X_list) == 0:
            raise RuntimeError('No basis was generated.')
        elif len(X_list) == 1:
            X = X_list[0]
        else:
            X = numpy.concatenate(X_list, axis=1)

        return X
