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

            # Incldue extra options
            traceinv_options.update(imate_options)

            # Create nterpolation object (only for traceinv)
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

        if exponent == 0:
            trace, _ = imate.trace(self.K, exponent=exponent)

        elif exponent == 1:
            trace, _ = imate.trace(self.K, exponent=exponent)

            if eta != 0:
                trace += eta * self.K.shape[0]

        elif exponent == 2:

            if eta == 0:
                trace, _ = imate.trace(self.K, exponent=exponent)
            else:
                trace_K, _ = imate.trace(self.K, exponent=1)
                trace_K2, _ = imate.trace(self.K, exponent=2)

                trace = trace_K2 + 2.0*eta * trace_K + eta**2 * self.K.shape[0]

        elif self.imate_method == 'eigenvalue':

            # Eigenvalues of mixed correlation K + eta*I
            Kn_eigenvalues = self.K_eigenvalues + eta

            # Using eigenvalues only. Here, self.K will not be used.
            trace, _ = imate.trace(self.K, method=self.imate_method,
                                   eigenvalues=Kn_eigenvalues,
                                   exponent=exponent, symmetric=True,
                                   **self.imate_options)

        elif self.imate_method == 'slq':

            # Passing the affine matrix function
            trace, _ = imate.trace(self.K_afm, parameters=eta,
                                   exponent=exponent, symmetric=True,
                                   **self.imate_options)

        else:
            raise ValueError('Existing methods are "exact", "eigenvalue", ' +
                             'and "slq".')

        return trace

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
            trace = self.interpolate_traceinv.interpolate(eta)

        elif self.imate_method == 'eigenvalue':

            # Eigenvalues of mixed correlation K + eta*I
            Kn_eigenvalues = self.K_eigenvalues + eta

            # Using eigenvalues only. Here, self.K will not be used.
            trace, _ = imate.traceinv(self.K, method=self.imate_method,
                                      eigenvalues=Kn_eigenvalues,
                                      exponent=exponent, symmetric=True,
                                      **self.imate_options)

        elif self.imate_method == 'cholesky':

            # Form the mixed covariance
            Kn = self.K + eta * self.I

            # Calling cholesky method
            trace, _ = imate.traceinv(Kn, method=self.imate_method,
                                      exponent=exponent,
                                      **self.imate_options)

        elif self.imate_method == 'hutchinson':

            # Form the mixed covariance
            Kn = self.K + eta * self.I

            # Calling cholesky method
            trace, _ = imate.traceinv(Kn, method=self.imate_method,
                                      exponent=exponent,
                                      assume_matrix='sym_pos',
                                      **self.imate_options)

        elif self.imate_method == 'slq':

            # Passing the affine matrix function
            trace, _ = imate.traceinv(self.K_afm, parameters=eta,
                                      exponent=exponent, symmetric=True,
                                      **self.imate_options)

        else:
            raise ValueError('Existing methods are "eigenvalue", "cholesky,"' +
                             '"hutchinson", and "slq".')

        return trace

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
                                      exponent=exponent, symmetric=True,
                                      **self.imate_options)

        elif self.imate_method == 'cholesky' or \
                self.imate_method == 'hutchinson':

            # Note: hutchinson method does not exists for logdet. So, we use
            # the cholesky method instead.

            # Form the mixed covariance
            Kn = self.K + eta * self.I

            # Calling cholesky method
            logdet_, _ = imate.logdet(Kn, method=self.imate_method,
                                      exponent=exponent, **self.imate_options)

        elif self.imate_method == 'slq':

            # Passing the affine matrix function
            logdet_, _ = imate.logdet(self.K_afm, parameters=eta,
                                      exponent=exponent, symmetric=True,
                                      **self.imate_options)

        else:
            raise ValueError('Existing methods are "eigenvalue", "cholesky",' +
                             ' and "slq".')

        return logdet_

    # =====
    # solve
    # =====

    def solve(self, eta, Y):
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

        Kn = self.K + eta*self.I
        X = linear_solver(Kn, Y, assume_matrix='sym_pos')

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

        if not isinstance(exponent, int):
            raise ValueError('"exponent" should be an integer.')
        elif exponent < 0:
            raise ValueError('"exponent" should be a non-negative integer.')

        y = numpy.zeros_like(x)

        for i in range(exponent):
            y += self.K.dot(x)
            if eta != 0:
                y += eta * x

        return y
