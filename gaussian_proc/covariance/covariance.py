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
from scipy.sparse import isspmatrix
from ..correlation import Correlation
from ._mixed_correlation import MixedCorrelation


# ==========
# Covariance
# ==========

class Covariance(object):
    """
    """

    # ====
    # init
    # ====

    def __init__(
            self,
            cor,
            sigma=None,
            sigma0=None,
            imate_method='cholesky',
            imate_options={},
            interpolate=False,
            tol=1e-8):
        """
        """

        self._check_arguments(cor, sigma, sigma0, tol)

        # Set attributes
        self.cor = cor
        self.sigma = sigma
        self.sigma0 = sigma0
        self.tol = tol

        # correlation matrix
        self.K = self.cor.get_matrix()

        # Mixed correlation (K + eta I)
        self.K_mixed = MixedCorrelation(self.K, interpolate=interpolate,
                                        imate_method=imate_method,
                                        imate_options=imate_options)

    # ===============
    # Check arguments
    # ===============

    def _check_arguments(self, cor, sigma, sigma0, tol):
        """
        """

        # Check tol
        if not isinstance(tol, float):
            raise TypeError('"tol" should be a float number.')
        elif tol < 0.0:
            raise ValueError('"tol" should be non-negative.')

        # Check cor
        if cor is None:
            raise ValueError('"cor" cannot be None.')

        elif not isinstance(cor, numpy.ndarray) and \
                not isspmatrix(cor) and \
                not isinstance(cor, Correlation):
            raise TypeError('"cor" should be either a "numpy.ndarray" ' +
                            'matrix or an instance of "Correlation" class.')

        if isinstance(cor, numpy.ndarray):
            if cor.ndim != 2:
                raise ValueError('"cor" should be a 2D matrix.')

            elif cor.shape[0] != cor.shape[1]:
                raise ValueError('"cor" should be a square matrix.')

            not_correlation = False
            for i in range(cor.shape[0]):
                if cor[i, i] != 1.0:
                    not_correlation = True
                    break

            if not_correlation:
                raise ValueError('Diagonal elements of "cor" should be "1".')

        # Check sigma
        if sigma is not None:
            if not isinstance(sigma, int) and isinstance(sigma, float):
                raise TypeError('"sigma" should be a float type.')
            elif sigma < 0.0:
                raise ValueError('"sigma" cannot be negative.')

        # Check sigma0
        if sigma0 is not None:
            if not isinstance(sigma0, int) and isinstance(sigma0, float):
                raise TypeError('"sigma0" should be a float type.')
            elif sigma0 < 0.0:
                raise ValueError('"sigma0" cannot be negative.')

    # =====
    # trace
    # =====

    def trace(self, sigma, sigma0, exponent=1):
        """
        Computes

        .. math::

            \\mathrm{trace} (\\sigma^2 \\mathbf{K} + \\sigma_0^2 \\mathbf{I}),

        where :math:`\\mathbf{I}` is the identity matrix and :math:`\\sigma`
        and :math:`\\sigma_0` are real numbers.
        """

        if numpy.abs(sigma) < self.tol:

            # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
            n = self.K_mixed.get_matrix_size()
            trace_ = (sigma0**(2.0*exponent)) * n
        else:
            eta = (sigma0 / sigma)**2
            trace_ = sigma**(2.0*exponent) * self.K_mixed.trace(eta, exponent)

        return trace_

    # ========
    # traceinv
    # ========

    def traceinv(self, sigma, sigma0, exponent=1):
        """
        Computes

        .. math::

            \\mathrm{trace} (\\sigma^2 \\mathbf{K} +
            \\sigma_0^2 \\mathbf{I})^{-1},

        where :math:`\\mathbf{I}` is the identity matrix and :math:`\\sigma`
        and :math:`\\sigma_0` are real numbers.
        """

        if numpy.abs(sigma) < self.tol:

            # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
            n = self.K_mixed.get_matrix_size()
            traceinv_ = n / (sigma0**(2.0*exponent))

        else:
            eta = (sigma0 / sigma)**2
            traceinv_ = self.K_mixed.traceinv(eta, exponent) / \
                (sigma**(2.0*exponent))

        return traceinv_

    # ======
    # logdet
    # ======

    def logdet(self, sigma, sigma0, exponent=1):
        """
        Computes

        .. math::

            \\mathrm{det} (\\sigma^2 \\mathbf{K} + \\sigma_0^2 \\mathbf{I}),

        where :math:`\\mathbf{I}` is the identity matrix and :math:`\\sigma`
        and :math:`\\sigma_0^2` are real numbers.
        """

        n = self.K_mixed.get_matrix_size()
        if numpy.abs(sigma) < self.tol:

            # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
            logdet_ = (2.0*exponent*n) * numpy.log(sigma0)

        else:
            eta = (sigma0 / sigma)**2
            logdet_ = (2.0*exponent*n) * numpy.log(sigma) + \
                self.K_mixed.logdet(eta, exponent)

        return logdet_

    # =====
    # solve
    # =====

    def solve(self, sigma, sigma0, Y, exponent=1):
        """
        Solves the linear system

        .. math::

            (\\sigma^2 \\mathbf{K} + \\sigma_0^2 \\mathbf{I}) \\mathbf{X}
            = \\mathbf{Y},

        where:

        * :math:`\\mathbf{Y}` is the given right hand side matrix,
        * :math:`\\mathbf{X}` is the solution (unknown) matrix,
        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`\\sigma` and :math:`\\sigma_0` are real numbers.
        """

        if numpy.abs(sigma) < self.tol:

            # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
            X = Y / (sigma0**(2*exponent))

        else:
            eta = (sigma0 / sigma)**2
            X = self.K_mixed.solve(eta, Y, exponent) / (sigma**(2*exponent))

        return X

    # ===
    # dot
    # ===

    def dot(self, sigma, sigma0, x, exponent=1):
        """
        Matrix-vector multiplication:

        .. math::

            \\boldsymbol{y} = (\\sigma^2 \\mathbf{K} +
            \\sigma_0^2 \\mathbf{I})^{q}
            \\boldsymbol{x}

        where:

        * :math:`\\boldsymbol{x}` is the given vector,
        * :math:`\\boldsymbol{y}` is the product vector,
        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`\\sigma` and :math:`\\sigma_0` are real numbers.
        * :math:`p`is a non-negative integer.
        """

        if exponent == 0:
            # Matrix is identity
            y = x

        else:
            if numpy.abs(sigma) < self.tol:

                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                y = sigma0**(2.0*exponent) * x

            else:
                eta = (sigma0 / sigma)**2
                y = (sigma**(2.0*exponent)) * \
                    self.K_mixed.dot(eta, x, exponent)

        return y
