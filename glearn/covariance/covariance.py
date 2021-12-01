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
import scipy
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

        # Options for imate
        self.imate_method = imate_method
        self.imate_options = imate_options

        # Mixed correlation (K + eta I)
        self.mixed_cor = MixedCorrelation(self.cor, interpolate=interpolate,
                                          imate_method=self.imate_method,
                                          imate_options=self.imate_options)

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
    # set sigmas
    # ==========

    def set_sigmas(self, sigma, sigma0):
        """
        After training, when optimal sigma and sigma0 is obtained, this
        function stores sigma and sigma0 as attributes of the class.
        """

        if sigma is None:
            raise ValueError('"sigma" cannot be None.')
        if sigma0 is None:
            raise ValueError('"sigma0" cannot be None.')

        self.sigma = sigma
        self.sigma0 = sigma0

        # Set eta for mixed_cor object
        self.mixed_cor.set_eta(self.sigma, self.sigma0)

    # ==========
    # get sigmas
    # ==========

    def get_sigmas(self, sigma=None, sigma0=None):
        """
        Returns sigma and sigma0. If the inputs are None, the object attributes
        are used.

        After training, when optimal sigma and sigma0 are obtained and set as
        the attributes of this class, the next calls to other functions like
        solve, trace, traceinv, etc, should use the optimal sigma and sigma0.
        Thus, we will call these functions without specifying sigma, and sigma0
        and this function returns the sigma and sigma0 that are stored as
        attributes.
        """

        # Get sigma
        if sigma is None:
            if self.sigma is None:
                raise ValueError('"sigma" cannot be None.')
            else:
                sigma = self.sigma

        # Get sigma0
        if sigma0 is None:
            if self.sigma0 is None:
                raise ValueError('"sigma0" cannot be None.')
            else:
                sigma0 = self.sigma0

        return sigma, sigma0

    # ==========
    # get matrix
    # ==========

    def get_matrix(
            self,
            sigma=None,
            sigma0=None,
            scale=None,
            derivative=[]):
        """
        Get the matrix as a numpy array of scipy sparse array.
        """

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if sigma < self.tol:

            if len(derivative) == 0:
                # Return scalar multiple of identity matrix
                S = sigma0**2 * self.mixed_cor.I
            else:
                # Return zero matrix
                n = self.mixed_cor.get_matrix_size()
                if self.cor.sparse:
                    S = scipy.sparse.csr_matrix((n, n))
                else:
                    S = numpy.zeros((n, n), dtype=float)

        else:

            eta = (sigma0 / sigma)**2
            Kn = self.mixed_cor.get_matrix(eta, scale, derivative)
            S = sigma**2 * Kn

        return S

    # =====
    # trace
    # =====

    def trace(
            self,
            sigma=None,
            sigma0=None,
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

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

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
            sigma=None,
            sigma0=None,
            B=None,
            C=None,
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

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if (B is None) and (C is not None):
            raise ValueError('When "C" is given, "B" should also be given.')

        if (exponent > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "exponent" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and exponent == 0:
            # Matrix is zero.
            traceinv_ = numpy.nan

        elif exponent == 0:
            # Matrix is identity, derivative is zero.
            if B is None:
                # B is identity
                n = self.mixed_cor.get_matrix_size()
                traceinv_ = n
            else:
                # B is not identity.
                if C is None:
                    traceinv_ = imate.trace(B, method='exact')
                else:
                    # C is not identity. Compute trace of C*B
                    if isspmatrix(C):
                        traceinv_ = numpy.sum(C.multiply(B.T).data)
                    else:
                        traceinv_ = numpy.sum(numpy.multiply(C, B.T))

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
                    # B is not identity
                    if C is None:
                        traceinv_ = imate.trace(B, method='exact') / \
                                (sigma0**(2.0*exponent))
                    else:
                        # C is not indentity. Compute trace of C*B devided by
                        # sigma0**4 (becase when we have C, there are to
                        # matrix A).
                        if isspmatrix(C):
                            traceinv_ = numpy.sum(C.multiply(B.T).data) / \
                                    (sigma0**(4.0*exponent))
                        else:
                            traceinv_ = numpy.sum(numpy.multiply(C, B.T)) / \
                                    (sigma0**(4.0*exponent))

        else:
            # Derivative eliminates sigma0^2*I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            traceinv_ = self.mixed_cor.traceinv(
                    eta, B, C, scale, exponent, derivative, imate_method)
            if C is None:
                traceinv_ /= sigma**(2.0*exponent)
            else:
                # When C is given, there are two A matrices (C*Ainv*B*Ainv)
                traceinv_ /= sigma**(4.0*exponent)

        return traceinv_

    # ======
    # logdet
    # ======

    def logdet(
            self,
            sigma=None,
            sigma0=None,
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

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

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
            Y,
            sigma=None,
            sigma0=None,
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

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

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
            X = numpy.copy(Y)

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
                    Y, eta, scale, exponent, derivative) / \
                (sigma**(2*exponent))

        return X

    # ===
    # dot
    # ===

    def dot(
            self,
            x,
            sigma=None,
            sigma0=None,
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

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

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
                self.mixed_cor.dot(x, eta, scale, exponent, derivative)

        return y

    # ===============
    # auto covariance
    # ===============

    def auto_covariance(self, test_points):
        """
        Computes the auto-covariance between the training points and
        themselves.
        """

        if self.sigma is None:
            raise RuntimeError('"sigma" cannot be None to create auto ' +
                               'covariance.')

        auto_cor = self.cor.auto_correlation(test_points)
        auto_cov = (self.sigma**2) * auto_cor

        return auto_cov

    # ================
    # cross covariance
    # ================

    def cross_covariance(self, test_points):
        """
        Computes the cross-covariance between the training points (points
        which this object is initialized with), and a given set of test points.
        This matrix is rectangular.
        """

        if self.sigma is None:
            raise RuntimeError('"sigma" cannot be None to create cross ' +
                               'covariance.')

        cross_cor = self.cor.cross_correlation(test_points)
        cross_cov = (self.sigma**2) * cross_cor

        return cross_cov
