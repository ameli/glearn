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

    def __init__(self, X, b=None, B=None):
        """
        """

        self._check_arguments(X, b, B)

        self.X = X    # Design matrix
        self.b = b    # Prior mean of beta
        self.B = B    # Prior covariance of beta

        # Precision of the prior of beta
        if self.B is not None:
            self.Binv = numpy.linalg.inv(self.B)
        else:
            # When B is None, we assume it is infinity. Hence, the precision
            # matrix (inverse of covariance) will be zero matrix.
            m = self.X.shape[1]
            self.Binv = numpy.zeros((m, m), dtype=float)

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
            b=None,
            B=None):
        """
        An alternative constructor used when the design matrix ``X`` is not
        known. This method designs the design matrix ``X`` and returns a class
        object with the computed ``X``.
        """

        # Generate design matrix X
        X = LinearModel.generate_design_matrix(points, polynomial_degree,
                                               trigonometric_coeff,
                                               hyperbolic_coeff)

        return cls(X, b, B)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, X, b, B):
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

        # Check b
        if b is not None:

            if numpy.isscalar(b) and (X.ndim != 1 or X.shape[1] != 1):
                raise ValueError('"b" should be a vector of the same size as' +
                                 'the number of columns of the design matrix' +
                                 '"X".')
            elif not isinstance(b, numpy.ndarray):
                raise TypeError('"b" should be either a scalar (if the' +
                                'design matrix is a column vector) or a row' +
                                'vector of "numpy.ndarray" type.')
            elif b.size != X.shape[1]:
                raise ValueError('"b" should have the same size as the ' +
                                 'number of columns of the design matrix ' +
                                 '"X", which is %d.' % X.shape[1])

        # Check B
        if B is not None:

            if b is None:
                raise ValueError('When "B" is given, "b" cannot be None.')
            elif numpy.isscalar(b) and not numpy.isscalar(B):
                raise ValueError('When "b" is a scalar, "B" should also be a' +
                                 'scalar.')
            elif not isinstance(B, numpy.ndarray):
                raise TypeError('"B" should be a "numpy.ndarray" type.')

            elif B.shape != (b.size, b.size):
                raise ValueError('"B" should be a square matrix with the' +
                                 'same number of columns/rows as the size of' +
                                 'vector "b".')

    # ======================
    # check design arguments
    # ======================

    @staticmethod
    def _check_design_arguments(
            points,
            polynomial_degree,
            trigonometric_coeff,
            hyperbolic_coeff):
        """
        """

        # Check points
        if points is None:
            raise ValueError('"points" cannot be "None".')

        elif not isinstance(points, numpy.ndarray):
            raise TypeError('"points" should be a "numpy.ndarray" type.')

        # Check at least one of polynomial, trigonometric or hyperbolic given.
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

        return trigonometric_coeff, hyperbolic_coeff

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

        trigonometric_coeff, hyperbolic_coeff = \
            LinearModel._check_design_arguments(
                    points, polynomial_degree, trigonometric_coeff,
                    hyperbolic_coeff)

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
            tri_size = trigonometric_coeff.size
            X_trigonometric = numpy.empty((n, 2*dimension*tri_size))

            for i in range(tri_size):
                for j in range(dimension):
                    X_trigonometric[:, 2*dimension*i + 2*j+0] = numpy.sin(
                            points[:, j] * trigonometric_coeff[i])
                    X_trigonometric[:, 2*dimension*i + 2*j+1] = numpy.cos(
                            points[:, j] * trigonometric_coeff[i])

            # append to the output list
            X_list.append(X_trigonometric)

        # Hyperbolic basis functions
        if hyperbolic_coeff is not None:
            hyp_size = hyperbolic_coeff.size
            X_hyperbolic = numpy.empty((n, 2*dimension*hyp_size))

            for i in range(hyp_size):
                for j in range(dimension):
                    X_hyperbolic[:, 2*dimension*i + 2*j+0] = numpy.sinh(
                            points[:, j] * hyperbolic_coeff[i])
                    X_hyperbolic[:, 2*dimension*i + 2*j+1] = numpy.cosh(
                            points[:, j] * hyperbolic_coeff[i])

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
