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
import imate


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

        # Mixed correlation (K + eta I)
        self.mixed_cor = MixedCorrelation(self.cor, interpolate=interpolate,
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

    # =========
    # set scale
    # =========

    def set_scale(self, scale):
        """
        Sets the scale attribute of coreelation matrix.
        """

        self.mixed_cor.set_scale(scale)

    # =========
    # get scale
    # =========

    def get_scale(self):
        """
        Returns distance scale of self.mixed_cor.cor object.
        """

        return self.mixed_cor.get_scale()

    # ==========
    # get matrix
    # ==========

    def get_matrix(
            self,
            sigma,
            sigma0,
            scale=None,
            derivative=[]):
        """
        Get the matrix as a numpy array of scipy sparse array.
        """

        eta = (sigma0 / sigma)**2
        Kn = self.mixed_cor.get_matrix(eta, scale, derivative)

        return sigma**2 * Kn

    # =====
    # trace
    # =====

    def trace(
            self,
            sigma,
            sigma0,
            scale=None,
            exponent=1,
            derivative=[],
            imate_method=None):
        """
        Computes

        .. math::

            \\mathrm{trace} \\frac{\\partial^q}{\\partial \\theta^q}
            (\\sigma^2 \\mathbf{K} + \\sigma_0^2 \\mathbf{I})^{p},

        where

        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`p`is a non-negative integer.
        * :math:`\\sigma` and :math:`\\sigma_0` are real numbers.
        * :math:`\\theta` is correlation scale parameter.
        * :math:`q` is the order of the derivative.
        """

        if (exponent > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "exponent" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and exponent == 0:
            # Matrix is zero.
            trace_ = 0.0

        elif exponent == 0:
            # Matrix is identity.
            n = self.mixed_cor.get_matrix_size()
            trace_ = n

        if numpy.abs(sigma) < self.tol:

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                trace_ = 0.0
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                n = self.mixed_cor.get_matrix_size()
                trace_ = (sigma0**(2.0*exponent)) * n

        else:
            # Derivative eliminates sigma0^2 I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            trace_ = sigma**(2.0*exponent) * self.mixed_cor.trace(
                    eta, scale, exponent, derivative, imate_method)

        return trace_

    # ========
    # traceinv
    # ========

    def traceinv(
            self,
            sigma,
            sigma0,
            B=None,
            scale=None,
            exponent=1,
            derivative=[],
            imate_method=None):
        """
        Computes

        .. math::

            \\mathrm{trace} \\left( \\frac{\\partial^q}{\\partial \\theta^q}
            (\\sigma^2 \\mathbf{K} + \\sigma_0^2 \\mathbf{I})^{-p} \\mathbf{B}
            \\right)

        where

        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`p`is a non-negative integer.
        * :math:`\\sigma` and :math:`\\sigma_0` are real numbers.
        * :math:`\\theta` is correlation scale parameter.
        * :math:`q` is the order of the derivative.
        * :math:`\\mathbf{B}` is a matrix. If set to None, identity matrix is
          assumed.
        """

        if (exponent > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "exponent" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and exponent == 0:
            # Matrix is zero.
            traceinv_ = numpy.nan

        elif exponent == 0:
            # Matrix is identity.
            if B is None:
                # B is identity
                n = self.mixed_cor.get_matrix_size()
                traceinv_ = n
            else:
                traceinv_ = imate.trace(B, method='exact')

        elif numpy.abs(sigma) < self.tol:

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                traceinv_ = numpy.nan
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                if B is None:
                    # B is identity
                    n = self.mixed_cor.get_matrix_size()
                    traceinv_ = n / (sigma0**(2.0*exponent))
                else:
                    traceinv_ = imate.trace(B, method='exact') \
                            / (sigma0**(2.0*exponent))

        else:
            # Derivative eliminates sigma0^2*I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            traceinv_ = self.mixed_cor.traceinv(
                    eta, B, scale, exponent, derivative, imate_method) / \
                (sigma**(2.0*exponent))

        return traceinv_

    # ======
    # logdet
    # ======

    def logdet(
            self,
            sigma,
            sigma0,
            scale=None,
            exponent=1,
            derivative=[],
            imate_method=None):
        """
        Computes

        .. math::

            \\mathrm{det} \\frac{\\partial^q}{\\partial \\theta^q}
            (\\sigma^2 \\mathbf{K} + \\sigma_0^2 \\mathbf{I})^{p},

        where

        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`p`is a non-negative integer.
        * :math:`\\sigma` and :math:`\\sigma_0` are real numbers.
        * :math:`\\theta` is correlation scale parameter.
        * :math:`q` is the order of the derivative.
        """

        if (exponent > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "exponent" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and exponent == 0:
            # Matrix is zero.
            logdet_ = -numpy.inf

        elif exponent == 0:
            # Matrix is identity.
            logdet_ = 0.0

        elif numpy.abs(sigma) < self.tol:

            n = self.mixed_cor.get_matrix_size()

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                logdet_ = -numpy.inf
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                logdet_ = (2.0*exponent*n) * numpy.log(sigma0)

        else:
            n = self.mixed_cor.get_matrix_size()

            # Derivative eliminates sigma0^2 I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            logdet_ = (2.0*exponent*n) * numpy.log(sigma) + \
                self.mixed_cor.logdet(eta, scale, exponent, derivative,
                                      imate_method)

        return logdet_

    # =====
    # solve
    # =====

    def solve(
            self,
            sigma,
            sigma0,
            Y,
            scale=None,
            exponent=1,
            derivative=[]):
        """
        Solves the linear system

        .. math::

            \\frac{\\partial^q}{\\partial \\theta^q}
            (\\sigma^2 \\mathbf{K} + \\sigma_0^2 \\mathbf{I})^{p} \\mathbf{X}
            = \\mathbf{Y},

        where:

        * :math:`\\mathbf{Y}` is the given right hand side matrix,
        * :math:`\\mathbf{X}` is the solution (unknown) matrix,
        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`p`is a non-negative integer.
        * :math:`\\sigma` and :math:`\\sigma_0` are real numbers.
        * :math:`\\theta` is correlation scale parameter.
        * :math:`q` is the order of the derivative.
        """

        if (exponent > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "exponent" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and exponent == 0:
            # Matrix is zero, hence has no inverse.
            X = numpy.zeros_like(Y)
            X[:] = numpy.nan

        elif exponent == 0:
            # Matrix is identity.
            X = Y.copy()

        elif numpy.abs(sigma) < self.tol:

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                X = numpy.zeros_like(Y)
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                X = Y / (sigma0**(2*exponent))

        else:
            # Derivative eliminates sigma0^2 I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            X = self.mixed_cor.solve(
                    eta, Y, scale, exponent, derivative) / \
                (sigma**(2*exponent))

        return X

    # ===
    # dot
    # ===

    def dot(
            self,
            sigma,
            sigma0,
            x,
            scale=None,
            exponent=1,
            derivative=[]):
        """
        Matrix-vector multiplication:

        .. math::

            \\boldsymbol{y} = \\frac{\\partial^q}{\\partial \\theta^q}
            (\\sigma^2 \\mathbf{K}(\\theta) + \\sigma_0^2 \\mathbf{I})^{p}
            \\boldsymbol{x}

        where:

        * :math:`\\boldsymbol{x}` is the given vector,
        * :math:`\\boldsymbol{y}` is the product vector,
        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`\\sigma` and :math:`\\sigma_0` are real numbers.
        * :math:`p`is a non-negative integer.
        * :math:`\\theta` is correlation scale parameter.
        * :math:`q` is the order of the derivative.
        """

        if (exponent > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "exponent" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif exponent == 0 and len(derivative) > 0:
            # Matrix is zero.
            y = numpy.zeros_like(x)

        elif exponent == 0:
            # Matrix is identity.
            y = x.copy()

        elif numpy.abs(sigma) < self.tol:

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                y = numpy.zeros_like(x)
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                y = sigma0**(2.0*exponent) * x

        else:
            # Derivative eliminates sigma0^2 I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            y = (sigma**(2.0*exponent)) * \
                self.mixed_cor.dot(eta, x, scale, exponent, derivative)

        return y
