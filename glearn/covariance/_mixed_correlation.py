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
import imate
from ._linear_solver import linear_solver
from .._utilities.timer import Timer


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

        # Set attributes
        self.cor = cor
        self.interpolate = interpolate
        self.interpolant_points = interpolant_points
        self.eta = None

        # Options for imate
        self.imate_method = imate_method
        self.imate_options = imate_options
        self.imate_info = {}

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

        # Elapsed time of computing functions (all iterations combined)
        self.logdet_timer = Timer()
        self.traceinv_timer = Timer()
        self.solve_timer = Timer()

    # =================
    # update imate info
    # =================

    def _update_imate_info(self, info):
        """
        Extracts the output info from am imate call and sets some information
        such as GPU device info, etc.
        """

        if self.imate_info == {}:
            self.imate_info = {
                'device': info['device'],
                'solver': info['solver']
            }
            if 'convergence' in info:
                self.imate_info['min_num_samples'] = \
                    info['convergence']['min_num_samples'],
                self.imate_info['max_num_samples'] = \
                    info['convergence']['max_num_samples']
            else:
                self.imate_info['min_num_samples'] = 0
                self.imate_info['max_num_samples'] = 0

    # =========
    # set scale
    # =========

    def set_scale(self, scale):
        """
        Sets the scale attribute of correlation matrix.
        """

        # Setting scale attribute of self.cor object.
        self.cor.set_scale(scale)

    # =========
    # get scale
    # =========

    def get_scale(self):
        """
        Returns distance scale of self.cor object.
        """

        return self.cor.get_scale()

    # ===============
    # get matrix size
    # ===============

    def get_matrix_size(self):
        """
        Returns the size of the correlation matrix.
        """

        return self.cor.get_matrix_size()

    # =======
    # set eta
    # =======

    def set_eta(self, sigma, sigma0):
        """
        After training, when optimal sigma and sigma0 is obtained, this
        function stores eta as an attribute of the class.
        """

        if sigma is None:
            raise ValueError('"sigma" cannot be None.')
        if sigma0 is None:
            raise ValueError('"sigma0" cannot be None.')

        self.eta = (sigma0 / sigma)**2

    # =======
    # get eta
    # =======

    def get_eta(self, eta=None):
        """
        Returns eta. If the input is None, the object attribute is used.

        After training, when optimal sigma and sigma0 are obtained, their ratio
        squared (as eta) is set as the attributes of this class. On the next
        calls to other functions like solve, trace, traceinv, etc, they will
        use the optimal eta. Thus, we will call these functions without
        specifying eta.
        """

        if eta is None:
            if self.eta is None:
                raise ValueError('"eta" cannot be None.')
            else:
                eta = self.eta

        return eta

    # ==========
    # get matrix
    # ==========

    def get_matrix(
            self,
            eta=None,
            scale=None,
            derivative=[]):
        """
        Get the matrix as a numpy array of scipy sparse array.
        """

        # Get eta (if None, uses class attribute)
        eta = self.get_eta(eta)

        K = self.cor.get_matrix(scale, derivative)

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
            eta=None,
            scale=None,
            derivative=[]):
        """
        Returns the eigenvalues of mixed correlation.
        """

        # Get eta (if None, uses class attribute)
        eta = self.get_eta(eta)

        K_eigenvalues = self.cor.get_eigenvalues(scale, derivative)

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
            eta=None,
            scale=None,
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

        # Get eta (if None, uses class attribute)
        eta = self.get_eta(eta)

        # Overwriting imate method, if not given.
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
            K = self.cor.get_matrix(scale, derivative)

            if isinstance(exponent, (int, numpy.integer)) or \
                    exponent.is_integer() or imate_method == 'exact':

                # Convert float to int
                if isinstance(exponent, float) and exponent.is_integer():
                    exponent = int(exponent)

                # Using Newton binomial for (K + eta*I)*exponent
                trace_ = 0.0
                for q in range(int(exponent)+1):
                    Kq_trace, info = imate.trace(K, method='exact', gram=False,
                                                 exponent=(exponent-q))
                    self._update_imate_info(info)

                    trace_ += binom(exponent, q) * Kq_trace * (eta**q)

            elif imate_method == 'eigenvalue':

                # Eigenvalues of mixed correlation K + eta*I
                Kn_eigenvalues = self.get_eigenvalues(eta, scale, derivative)

                # Using eigenvalues only. Here, K will not be used.
                trace_, info = imate.trace(K, method=imate_method,
                                           eigenvalues=Kn_eigenvalues,
                                           exponent=exponent, gram=False,
                                           assume_matrix='sym',
                                           **self.imate_options)
                self._update_imate_info(info)

            elif imate_method == 'slq':

                # Get affine matrix function
                K_amf = self.cor.get_affine_matrix_function(scale, derivative)

                # Passing the affine matrix function
                trace_, info = imate.trace(K_amf, method=imate_method,
                                           parameters=eta, exponent=exponent,
                                           gram=False, **self.imate_options)
                self._update_imate_info(info)

            else:
                raise ValueError('Existing methods are "exact", ' +
                                 '"eigenvalue", and "slq".')

        return trace_

    # ========
    # traceinv
    # ========

    def traceinv(
            self,
            eta=None,
            B=None,
            C=None,
            scale=None,
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

        self.traceinv_timer.tic()

        # Get eta (if None, uses class attribute)
        eta = self.get_eta(eta)

        # Overwriting imate method, if not given.
        if imate_method is None:
            imate_method = self.imate_method

        # When B is given, only Cholesky and Hutchinson methods can be used.
        if B is not None:
            if imate_method == 'eigenvalue':
                # Change method to a similar non-stochastic method
                imate_method = 'cholesky'
            elif imate_method == 'slq':
                # Change method to a similar stochastic method
                imate_method = 'hutchinson'

        # When C is given, only Hutchinson method can be used.
        if C is not None:
            imate_method = 'hutchinson'

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
                n = self.cor.get_matrix_size()
                traceinv_ = n
            else:
                # B is not identity
                if C is None:
                    traceinv_, info = imate.trace(B, method='exact')
                    self._update_imate_info(info)
                else:
                    # C is not identity. Compute trace of C*B
                    if isspmatrix(C):
                        traceinv_ = numpy.sum(C.multiply(B.T).data)
                    else:
                        traceinv_ = numpy.sum(numpy.multiply(C, B.T))

        elif numpy.abs(eta) >= self.max_eta:
            if B is None:
                # B is identity
                traceinv_ = self.get_matrix_size() / (eta**exponent)
            else:
                # B is not identity
                if C is None:
                    traceinv_, info = imate.trace(B, method='exact') / \
                            (eta**exponent)
                    self._update_imate_info(info)
                else:
                    # C is not identity. Compute trace of C*B divided by
                    # eta**(2.0*exponent) since when C is given, there are two
                    # matrix A.
                    if isspmatrix(C):
                        traceinv_ = numpy.sum(C.multiply(B.T).data) / \
                                (eta**(2.0*exponent))

                    else:
                        traceinv_ = numpy.sum(numpy.multiply(C, B.T)) / \
                                (eta**(2.0*exponent))

        else:

            if self.interpolate:

                # Interpolate traceinv
                if B is None:
                    # B is identity
                    traceinv_ = self.interpolate_traceinv.interpolate(eta)
                else:
                    raise NotImplementedError('Interpolating traceinv of ' +
                                              'mixed correlation matrix ' +
                                              'when B (or C) is not ' +
                                              'identity, is not implemented.')

            elif imate_method == 'eigenvalue':

                # Get matrix
                K = self.cor.get_matrix(scale, derivative)

                # Eigenvalues of mixed correlation K + eta*I
                Kn_eigenvalues = self.get_eigenvalues(eta, scale, derivative)

                # Using eigenvalues only. Here, K will not be used.
                traceinv_, info = imate.traceinv(K, method=imate_method,
                                                 eigenvalues=Kn_eigenvalues,
                                                 exponent=exponent, gram=False,
                                                 assume_matrix='sym',
                                                 **self.imate_options)
                self._update_imate_info(info)

            elif imate_method == 'cholesky':

                # Form the mixed covariance
                Kn = self.get_matrix(eta, scale, derivative)

                # Calling cholesky method
                traceinv_, info = imate.traceinv(Kn, B, method=imate_method,
                                                 exponent=exponent, gram=False,
                                                 **self.imate_options)
                self._update_imate_info(info)

            elif imate_method == 'hutchinson':

                # Form the mixed correlation
                Kn = self.get_matrix(eta, scale, derivative)

                if len(derivative) > 0:
                    assume_matrix = 'sym'
                else:
                    assume_matrix = 'sym_pos'

                # Calling cholesky method
                traceinv_, info = imate.traceinv(Kn, B, C, method=imate_method,
                                                 exponent=exponent, gram=False,
                                                 assume_matrix=assume_matrix,
                                                 **self.imate_options)
                self._update_imate_info(info)

            elif imate_method == 'slq':

                # Get affine matrix function
                K_amf = self.cor.get_affine_matrix_function(scale, derivative)

                # Passing the affine matrix function
                traceinv_, info = imate.traceinv(K_amf, method=imate_method,
                                                 parameters=eta,
                                                 exponent=exponent, gram=False,
                                                 **self.imate_options)
                self._update_imate_info(info)

            else:
                raise ValueError('Existing methods are "eigenvalue", ' +
                                 '"cholesky", "hutchinson", and "slq".')

        self.traceinv_timer.toc()

        return traceinv_

    # ======
    # logdet
    # ======

    def logdet(
            self,
            eta=None,
            scale=None,
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

        self.logdet_timer.tic()

        # Get eta (if None, uses class attribute)
        eta = self.get_eta(eta)

        # Overwriting imate method, if not given.
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
                K = self.cor.get_matrix(scale, derivative)

                # Eigenvalues of mixed correlation K + eta*I
                Kn_eigenvalues = self.get_eigenvalues(eta, scale, derivative)

                # Using eigenvalues only. Here, K will not be used.
                logdet_, info = imate.logdet(K, method=imate_method,
                                             eigenvalues=Kn_eigenvalues,
                                             exponent=exponent, gram=False,
                                             assume_matrix='sym',
                                             **self.imate_options)
                self._update_imate_info(info)

            elif imate_method == 'cholesky':

                # Note: hutchinson method does not exists for logdet. So, we
                # use the cholesky method instead.

                # Form the mixed correlation
                Kn = self.get_matrix(eta, scale, derivative)

                # Calling cholesky method
                logdet_, info = imate.logdet(Kn, method='cholesky', gram=False,
                                             exponent=exponent,
                                             **self.imate_options)
                self._update_imate_info(info)

            elif imate_method == 'slq':

                # Get affine matrix function
                K_amf = self.cor.get_affine_matrix_function(scale, derivative)

                # Passing the affine matrix function
                logdet_, info = imate.logdet(K_amf, method=imate_method,
                                             parameters=eta, exponent=exponent,
                                             gram=False, **self.imate_options)
                self._update_imate_info(info)

            else:
                raise ValueError('Existing methods are "eigenvalue", ' +
                                 '"cholesky", and "slq".')

        self.logdet_timer.toc()

        return logdet_

    # =====
    # solve
    # =====

    def solve(
            self,
            Y,
            eta=None,
            scale=None,
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

        self.solve_timer.tic()

        # Get eta (if None, uses class attribute)
        eta = self.get_eta(eta)

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
            Kn = self.get_matrix(eta, scale, derivative)

            if len(derivative) > 0:
                assume_matrix = 'sym'
            else:
                assume_matrix = 'sym_pos'

            X = Y.copy()
            for i in range(exponent):
                X = linear_solver(Kn, X, assume_matrix=assume_matrix)

        self.solve_timer.toc()

        return X

    # ===
    # dot
    # ===

    def dot(
            self,
            x,
            eta=None,
            scale=None,
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

        # Get eta (if None, uses class attribute)
        eta = self.get_eta(eta)

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
            K = self.get_matrix(0.0, scale, derivative)

            for i in range(exponent):
                y = K.dot(x_copy)

                if (len(derivative) == 0) and (eta != 0):
                    y += eta * x_copy

                # Update x_copy for next iteration
                if i < exponent - 1:
                    x_copy = y.copy()

        return y
