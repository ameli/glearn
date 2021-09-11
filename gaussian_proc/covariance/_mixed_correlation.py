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
from scipy.special import binom
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

    def __init__(self, cor, interpolate=False, interpolant_points=None,
                 imate_method='cholesky', imate_options={}, tol=1e-16):
        """
        """

        self.cor = cor
        self.interpolate = interpolate
        self.interpolant_points = interpolant_points

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
        if self.cor.sparse:
            self.I = scipy.sparse.eye(self.get_matrix_size(),      # noqa: E741
                                      format='csr')
        else:
            self.I = numpy.eye(self.get_matrix_size())             # noqa: E741

        # Lower and upper bounds for eta
        self.min_eta = tol
        self.max_eta = 1.0/tol

    # ==================
    # set distance scale
    # ==================

    def set_distance_scale(self, distance_scale):
        """
        Sets the distance_scale attribute of coreelation matrix.
        """

        # Setting distance_scale attribute of self.cor object.
        self.cor.set_distance_scale(distance_scale)

    # ==================
    # get distance scale
    # ==================

    def get_distance_scale(self):
        """
        Returns distance scale of self.cor object.
        """

        return self.cor.distance_scale

    # ===============
    # get matrix size
    # ===============

    def get_matrix_size(self):
        """
        Returns the size of the correlation matrix.
        """

        return self.cor.get_matrix_size()

    # ==========
    # get matrix
    # ==========

    def get_matrix(
            self,
            eta,
            distance_scale=None,
            derivative=[]):
        """
        Get the matrix as a numpy array of scipy sparse array.
        """

        K = self.cor.get_matrix(distance_scale, derivative)

        # Form the mixed correlation
        if len(derivative) > 0:
            Kn = K
        else:
            if eta != 0.0:
                Kn = K + eta * self.I
            else:
                Kn = K

        return Kn

    # ===============
    # get eigenvalues
    # ===============

    def get_eigenvalues(
            self,
            eta,
            distance_scale=None,
            derivative=[]):
        """
        Returns the eigenvalues of mixed correlation.
        """

        K_eigenvalues = self.cor.get_eigenvalues(distance_scale, derivative)

        if len(derivative) > 0:
            Kn_eigenvalues = K_eigenvalues
        else:
            Kn_eigenvalues = K_eigenvalues + eta

        return Kn_eigenvalues

    # =====
    # trace
    # =====

    def trace(
            self,
            eta,
            distance_scale=None,
            exponent=1,
            derivative=[],
            imate_method=None):
        """
        Computes

        .. math::

            \\mathrm{trace} (\\mathbf{K} + \\eta \\mathbf{I}),

        where :math:`\\mathbf{I}` is the identity matrix and :math:`\\eta` is a
        real number.
        """

        # Overwriting imate method, if given.
        if imate_method is None:
            imate_method = self.imate_method

        if (exponent > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "exponent" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and exponent == 0:
            # Matrix is zero.
            trace_ = 0.0

        elif exponent == 0:
            # Matrix is identity.
            n = self.cor.get_matrix_size()
            trace_ = n

        elif numpy.abs(eta) >= self.max_eta:
            trace_ = eta * self.get_matrix_size()

        else:

            # Get matrix
            K = self.cor.get_matrix(distance_scale, derivative)

            if isinstance(exponent, (int, numpy.integer)) or \
                    exponent.is_integer() or imate_method == 'exact':

                # Convert float to int
                if isinstance(exponent, float) and exponent.is_integer():
                    exponent = int(exponent)

                # Using Newton binomial for (K + eta*I)*exponent
                trace_ = 0.0
                for q in range(int(exponent)+1):
                    Kq_trace, _ = imate.trace(K, method='exact', gram=False,
                                              exponent=(exponent-q))

                    trace_ += binom(exponent, q) * Kq_trace * (eta**q)

            elif imate_method == 'eigenvalue':

                # Eigenvalues of mixed correlation K + eta*I
                Kn_eigenvalues = self.get_eigenvalues(eta, distance_scale,
                                                      derivative)

                # Using eigenvalues only. Here, K will not be used.
                trace_, _ = imate.trace(K, method=imate_method,
                                        eigenvalues=Kn_eigenvalues,
                                        exponent=exponent, gram=False,
                                        assume_matrix='sym',
                                        **self.imate_options)

            elif imate_method == 'slq':

                # Get affine matrix function
                K_amf = self.cor.get_affine_matrix_function(distance_scale,
                                                            derivative)

                # Passing the affine matrix function
                trace_, _ = imate.trace(K_amf, method=imate_method,
                                        parameters=eta, exponent=exponent,
                                        gram=False, **self.imate_options)

            else:
                raise ValueError('Existing methods are "exact", ' +
                                 '"eigenvalue", and "slq".')

        return trace_

    # ========
    # traceinv
    # ========

    def traceinv(
            self,
            eta,
            B=None,
            distance_scale=None,
            exponent=1,
            derivative=[],
            imate_method=None):
        """
        Computes

        .. math::

            \\mathrm{trace} \\left( (\\mathbf{K} + \\eta \\mathbf{I})^{-1}
            \\mathbf{B} \\right)

        where :math:`\\mathbf{I}` is the identity matrix and :math:`\\eta` is a
        real number. If :math:`\\mathbf{B}` is set to None, identity matrix is
        assumed.
        """

        # Overwriting imate method, if given.
        if imate_method is None:
            imate_method = self.imate_method

        # When B is not None, eigenvalue and slq methods cannot be used.
        if (B is not None) and (imate_method in ['eigenvalue', 'slq']):
            raise NotImplementedError('Computing traceinv of mixed ' +
                                      'correlation matrix using "eigenvalue"' +
                                      'or "slq" methods, when B is not ' +
                                      'is not implemented. Use "cholesky" ' +
                                      'or "hutchinson" methods instead.')

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
                n = self.cor.get_matrix_size()
                traceinv_ = n
            else:
                traceinv_ = imate.trace(B, method='exact')

        elif numpy.abs(eta) >= self.max_eta:
            if B is None:
                # B is identity
                traceinv_ = self.get_matrix_size() / eta
            else:
                traceinv_ = imate.trace(B, method='exact') / eta

        else:

            if self.interpolate:

                # Interpolate traceinv
                if B is None:
                    # B is identity
                    traceinv_ = self.interpolate_traceinv.interpolate(eta)
                else:
                    raise NotImplementedError('Interpolating traceinv of ' +
                                              'mixed correlation matrix ' +
                                              'when B is not identity, is ' +
                                              'not implemented.')

            elif imate_method == 'eigenvalue':

                # Get matrix
                K = self.cor.get_matrix(distance_scale, derivative)

                # Eigenvalues of mixed correlation K + eta*I
                Kn_eigenvalues = self.get_eigenvalues(eta, distance_scale,
                                                      derivative)

                # Using eigenvalues only. Here, K will not be used.
                traceinv_, _ = imate.traceinv(K, method=imate_method,
                                              eigenvalues=Kn_eigenvalues,
                                              exponent=exponent, gram=False,
                                              assume_matrix='sym',
                                              **self.imate_options)

            elif imate_method == 'cholesky':

                # Form the mixed covariance
                Kn = self.get_matrix(eta, distance_scale, derivative)

                # Calling cholesky method
                traceinv_, _ = imate.traceinv(Kn, B, method=imate_method,
                                              exponent=exponent, gram=False,
                                              **self.imate_options)

            elif imate_method == 'hutchinson':

                # Form the mixed correlation
                Kn = self.get_matrix(eta, distance_scale, derivative)

                if len(derivative) > 0:
                    assume_matrix = 'sym'
                else:
                    assume_matrix = 'sym_pos'

                # Calling cholesky method
                traceinv_, _ = imate.traceinv(Kn, B, method=imate_method,
                                              exponent=exponent, gram=False,
                                              assume_matrix=assume_matrix,
                                              **self.imate_options)

            elif imate_method == 'slq':

                # Get affine matrix function
                K_amf = self.cor.get_affine_matrix_function(distance_scale,
                                                            derivative)

                # Passing the affine matrix function
                traceinv_, _ = imate.traceinv(K_amf, method=imate_method,
                                              parameters=eta,
                                              exponent=exponent, gram=False,
                                              **self.imate_options)

            else:
                raise ValueError('Existing methods are "eigenvalue", ' +
                                 '"cholesky", "hutchinson", and "slq".')

        return traceinv_

    # ======
    # logdet
    # ======

    def logdet(
            self,
            eta,
            distance_scale=None,
            exponent=1,
            derivative=[],
            imate_method=None):
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

        # Overwriting imate method, if given.
        if imate_method is None:
            imate_method = self.imate_method

        # Logdet does not have hutchinson method. So, pass to cholesky instead.
        if imate_method == 'hutchinson':
            imate_method = 'cholesky'

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

        elif numpy.abs(eta) >= self.max_eta:
            logdet_ = self.get_matrix_size() * numpy.log(eta)

        else:

            if imate_method == 'eigenvalue':

                # Get matrix
                K = self.cor.get_matrix(distance_scale, derivative)

                # Eigenvalues of mixed correlation K + eta*I
                Kn_eigenvalues = self.get_eigenvalues(eta, distance_scale,
                                                      derivative)

                # Using eigenvalues only. Here, K will not be used.
                logdet_, _ = imate.logdet(K, method=imate_method,
                                          eigenvalues=Kn_eigenvalues,
                                          exponent=exponent, gram=False,
                                          assume_matrix='sym',
                                          **self.imate_options)

            elif imate_method == 'cholesky':

                # Note: hutchinson method does not exists for logdet. So, we
                # use the cholesky method instead.

                # Form the mixed correlation
                Kn = self.get_matrix(eta, distance_scale, derivative)

                # Calling cholesky method
                logdet_, _ = imate.logdet(Kn, method='cholesky', gram=False,
                                          exponent=exponent,
                                          **self.imate_options)

            elif imate_method == 'slq':

                # Get affine matrix function
                K_amf = self.cor.get_affine_matrix_function(distance_scale,
                                                            derivative)

                # Passing the affine matrix function
                logdet_, _ = imate.logdet(K_amf, method=imate_method,
                                          parameters=eta, exponent=exponent,
                                          gram=False, **self.imate_options)

            else:
                raise ValueError('Existing methods are "eigenvalue", ' +
                                 '"cholesky", and "slq".')

        return logdet_

    # =====
    # solve
    # =====

    def solve(
            self,
            eta,
            Y,
            distance_scale=None,
            exponent=1,
            derivative=[]):
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

        elif numpy.abs(eta) >= self.max_eta:
            X = Y.copy() / eta

        else:
            # Get matrix
            Kn = self.get_matrix(eta, distance_scale, derivative)

            if len(derivative) > 0:
                assume_matrix = 'sym'
            else:
                assume_matrix = 'sym_pos'

            X = Y.copy()
            for i in range(exponent):
                X = linear_solver(Kn, X, assume_matrix=assume_matrix)

        return X

    # ===
    # dot
    # ===

    def dot(
            self,
            eta,
            x,
            distance_scale=None,
            exponent=1,
            derivative=[]):
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

        # Check exponent
        if not isinstance(exponent, (int, numpy.integer)):
            raise ValueError('"exponent" should be an integer.')
        elif exponent < 0:
            raise ValueError('"exponent" should be a non-negative integer.')

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

        elif numpy.abs(eta) >= self.max_eta:
            y = x.copy() * eta

        else:
            x_copy = x.copy()

            # Get matrix (K only, not K + eta * I)
            K = self.get_matrix(0.0, distance_scale, derivative)

            for i in range(exponent):
                y = K.dot(x_copy)

                if (len(derivative) == 0) and (eta != 0):
                    y += eta * x_copy

                # Update x_copy for next iteration
                if i < exponent - 1:
                    x_copy = y.copy()

        return y
