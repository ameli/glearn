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
import scipy.sparse
from scipy.sparse import isspmatrix
from scipy.special import binom
import scipy.linalg
import imate
from ._linear_solver import linear_solver


# =================
# Mixed Correlation
# =================

class MixedCorrelation(object):
    """
    A wrapper class for ``imate.AffineMatrixFunction``.
    """

    # ====
    # init
    # ====

    def __init__(self, K, interpolate=False, interpolant_points=None,
                 imate_method='cholesky', imate_options={}):
        """
        """

        self.K = K
        self.interpolate = interpolate
        self.interpolant_points = interpolant_points

        # Create affine matrix function (amf) object
        self.K_amf = imate.AffineMatrixFunction(K)

        # Options for imate
        self.imate_method = imate_method
        self.imate_options = imate_options

        # Interpolate traceinv
        self.interpolate_traceinv = None
        if self.interpolate:
            if self.interpolant_points is None:
                raise TypeError('When "interpolate" is set to "True", the ' +
                                '"interpolant_points" cannot be None.')

            traceinv_options = {
                'method': imate_method
            }

            # Include extra options
            traceinv_options.update(imate_options)

            # Create interpolation object (only for traceinv)
            self.interpolate_traceinv = imate.InterpolateTraceInv(
                    self.K, traceinv_options=traceinv_options)

        # Identity matrix
        if scipy.sparse.isspmatrix(self.K):
            self.I = scipy.sparse.eye(self.K.shape[0],             # noqa: E741
                                      format=self.K.format)
        else:
            self.I = numpy.eye(self.K.shape[0])                    # noqa: E741

        # Eigenvalues method
        self.K_eigenvalues = None
        if self.imate_method == 'eigenvalue':

            if isspmatrix(self.K):
                raise RuntimeError('When the correlation matrix is sparse, ' +
                                   'the "imate_method" cannot be set to ' +
                                   '"eigenvalue". You may set ' +
                                   '"imate_method" to "cholesky", "slq", or ' +
                                   '"hutchinson."')

            self.K_eigenvalues = scipy.linalg.eigh(
                    self.K, eigvals_only=True, check_finite=False)

    # ===============
    # get matrix size
    # ===============

    def get_matrix_size(self):
        """
        Returns the size of the correlation matrix.
        """

        return self.K.shape[0]

    # =====
    # trace
    # =====

    def trace(self, eta, exponent=1):
        """
        Computes

        .. math::

            \\mathrm{trace} (\\mathbf{K} + \\eta \\mathbf{I}),

        where :math:`\\mathbf{I}` is the identity matrix and :math:`\\eta` is a
        real number.
        """

        if isinstance(exponent, (int, numpy.integer)) or \
                exponent.is_integer() or self.imate_method == 'exact':

            # Convert float to int
            if isinstance(exponent, float) and exponent.is_integer():
                exponent = int(exponent)

            # Using Newton binomial for (K + eta*I)*exponent
            trace_ = 0.0
            for q in range(int(exponent)+1):
                Kq_trace, _ = imate.trace(self.K, method='exact',
                                          gram=False, exponent=(exponent-q))

                trace_ += binom(exponent, q) * Kq_trace * (eta**q)

        elif self.imate_method == 'eigenvalue':

            # Eigenvalues of mixed correlation K + eta*I
            Kn_eigenvalues = self.K_eigenvalues + eta

            # Using eigenvalues only. Here, self.K will not be used.
            trace_, _ = imate.trace(self.K, method=self.imate_method,
                                    eigenvalues=Kn_eigenvalues,
                                    exponent=exponent, gram=False,
                                    assume_matrix='sym', **self.imate_options)

        elif self.imate_method == 'slq':

            # Passing the affine matrix function
            trace_, _ = imate.trace(self.K_amf, method=self.imate_method,
                                    parameters=eta, exponent=exponent,
                                    gram=False, **self.imate_options)

        else:
            raise ValueError('Existing methods are "exact", "eigenvalue", ' +
                             'and "slq".')

        return trace_

    # ========
    # traceinv
    # ========

    def traceinv(self, eta, exponent=1):
        """
        Computes

        .. math::

            \\mathrm{trace} (\\mathbf{K} + \\eta \\mathbf{I})^{-1},

        where :math:`\\mathbf{I}` is the identity matrix and :math:`\\eta` is a
        real number.
        """

        if self.interpolate:

            # Interpolate traceinv
            traceinv_ = self.interpolate_traceinv.interpolate(eta)

        elif self.imate_method == 'eigenvalue':

            # Eigenvalues of mixed correlation K + eta*I
            Kn_eigenvalues = self.K_eigenvalues + eta

            # Using eigenvalues only. Here, self.K will not be used.
            traceinv_, _ = imate.traceinv(self.K, method=self.imate_method,
                                          eigenvalues=Kn_eigenvalues,
                                          exponent=exponent, gram=False,
                                          assume_matrix='sym',
                                          **self.imate_options)

        elif self.imate_method == 'cholesky':

            # Form the mixed covariance
            Kn = self.K + eta * self.I

            # Calling cholesky method
            traceinv_, _ = imate.traceinv(Kn, method=self.imate_method,
                                          exponent=exponent, gram=False,
                                          **self.imate_options)

        elif self.imate_method == 'hutchinson':

            # Form the mixed covariance
            Kn = self.K + eta * self.I

            # Calling cholesky method
            traceinv_, _ = imate.traceinv(Kn, method=self.imate_method,
                                          exponent=exponent, gram=False,
                                          assume_matrix='sym_pos',
                                          **self.imate_options)

        elif self.imate_method == 'slq':

            # Passing the affine matrix function
            traceinv_, _ = imate.traceinv(self.K_amf, method=self.imate_method,
                                          parameters=eta, exponent=exponent,
                                          gram=False, **self.imate_options)

        else:
            raise ValueError('Existing methods are "eigenvalue", "cholesky,"' +
                             '"hutchinson", and "slq".')

        return traceinv_

    # ======
    # logdet
    # ======

    def logdet(self, eta, exponent=1):
        """
        Computes

        .. math::

            \\mathrm{det} (\\mathbf{K} + \\eta \\mathbf{I}),

        where :math:`\\mathbf{I}` is the identity matrix and :math:`\\eta` is
        a real number.

        .. note::

            If ``self.imate_method`` is set to ``hutchinson``, since such
            method is not applicable to ``logdet()``, we use ``cholesky``
            instead.
        """

        if self.imate_method == 'eigenvalue':

            # Eigenvalues of mixed correlation K + eta*I
            Kn_eigenvalues = self.K_eigenvalues + eta

            # Using eigenvalues only. Here, self.K will not be used.
            logdet_, _ = imate.logdet(self.K, method=self.imate_method,
                                      eigenvalues=Kn_eigenvalues,
                                      exponent=exponent, gram=False,
                                      assume_matrix='sym',
                                      **self.imate_options)

        elif self.imate_method == 'cholesky' or \
                self.imate_method == 'hutchinson':

            # Note: hutchinson method does not exists for logdet. So, we use
            # the cholesky method instead.

            # Form the mixed covariance
            Kn = self.K + eta * self.I

            # Calling cholesky method
            logdet_, _ = imate.logdet(Kn, method='cholesky', gram=False,
                                      exponent=exponent, **self.imate_options)

        elif self.imate_method == 'slq':

            # Passing the affine matrix function
            logdet_, _ = imate.logdet(self.K_amf, method=self.imate_method,
                                      parameters=eta, exponent=exponent,
                                      gram=False, **self.imate_options)

        else:
            raise ValueError('Existing methods are "eigenvalue", "cholesky",' +
                             ' and "slq".')

        return logdet_

    # =====
    # solve
    # =====

    def solve(self, eta, Y, exponent=1):
        """
        Solves the linear system

        .. math::

            (\\mathbf{K} + \\eta \\mathbf{I}) \\mathbf{X} = \\mathbf{Y},

        where:

        * :math:`\\mathbf{Y}` is the given right hand side matrix,
        * :math:`\\mathbf{X}` is the solution (unknown) matrix,
        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`\\eta` is a real number.
        """

        X = Y.copy()
        Kn = self.K + eta*self.I

        for i in range(exponent):
            X = linear_solver(Kn, X, assume_matrix='sym_pos')

        return X

    # ===
    # dot
    # ===

    def dot(self, eta, x, exponent=1):
        """
        Matrix-vector multiplication:

        .. math::

            \\boldsymbol{y} = (\\mathbf{K} + \\eta \\mathbf{I})^{q}
            \\boldsymbol{x}

        where:

        * :math:`\\boldsymbol{x}` is the given vector,
        * :math:`\\boldsymbol{y}` is the product vector,
        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`\\eta` is a real number,
        * :math:`p`is a non-negative integer.
        """

        if not isinstance(exponent, (int, numpy.integer)):
            raise ValueError('"exponent" should be an integer.')
        elif exponent < 0:
            raise ValueError('"exponent" should be a non-negative integer.')

        if exponent == 0:
            # Matrix is identity
            y = x

        else:
            x_copy = x.copy()

            for i in range(exponent):
                y = self.K.dot(x_copy)
                if eta != 0:
                    y += eta * x_copy

                # Update x_copy for next iteration
                if i < exponent - 1:
                    x_copy = y.copy()

        return y
