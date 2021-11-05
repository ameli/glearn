# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import numpy
import imate
from ._base_likelihood import BaseLikelihood


# ===============
# Full Likelihood
# ===============

class FullLikelihood(BaseLikelihood):

    # Import plot-related methods of this class implemented in a separate file
    from ._full_likelihood_plots import plot

    # ====
    # init
    # ====

    def __init__(self, mean, cov, z, log_hyperparam=True):
        """
        Initialization.
        """

        # Super class constructor sets self.z, self.X, self.cov, self.mixed_cor
        super().__init__(mean, cov, z)

        # The index in hyperparam array where scale starts. In this class,
        # hyperparam is of the form [sigma, sigma0, scale], hence, scale
        # starts at index 2.
        self.scale_index = 2

        # Configuration
        self.hyperparam_tol = 1e-8

        if log_hyperparam:
            self.use_log_scale = True
        else:
            self.use_log_scale = False

        # Determine to compute traceinv (only for some of inner computations of
        # derivatives w.r.t scale) using direct inversion of matrices or with
        # Hutchinson method (a stochastic method).
        if self.cov.imate_method in ['hutchinson', 'slq']:
            # Use Hutchinson method (note: SLQ method cannot be used).
            self.stochastic_traceinv = True
        else:
            # For the rest of methods (like eigenvalue, cholesky, etc),
            # compute traceinv directly by matrix inversion.
            self.stochastic_traceinv = False

        # Store ell, its Jacobian and Hessian.
        self.ell = None
        self.ell_jacobian = None
        self.ell_hessian = None

        # Store hyperparam used to compute ell, its Jacobian and Hessian.
        self.ell_hyperparam = None
        self.ell_jacobian_hyperparam = None
        self.ell_hessian_hyperparam = None

        # Interval variables that are shared between class methods
        self.Y = None
        self.B = None
        self.Binv = None
        self.Mz = None
        self.MMz = None
        self.KMz = None
        self.MKMz = None
        self.trace_M = None
        self.Sinv = None
        self.KpSinv = None
        self.SpSinv = None

        # Hyperparameter which the interval variables in the above were
        # computed based upon it.
        self.Y_B_Mz_hyperparam = None
        self.MMz_KMz_MKMz_hyperparam = None
        self.trace_M_hyperparam = None
        self.Sinv_KpSinv_SpSinv_hyperparam = None

    # ====================
    # hyperparam to sigmas
    # ====================

    def _hyperparam_to_sigmas(self, hyperparam):
        """
        Sets sigma and sigma0 from hyperparam.
        """

        sigma = numpy.abs(hyperparam[0])
        sigma0 = numpy.abs(hyperparam[1])

        return sigma, sigma0

    # ===================
    # scale to hyperparam
    # ===================

    def _scale_to_hyperparam(self, scale):
        """
        Sets hyperparam from scale. scale is always given with no log-scale
        If self.use_log_eta is True, hyperparam is set as log10 of scale,
        otherwise, just as scale.
        """

        # If logscale is used, output hyperparam is log of scale.
        if self.use_log_scale:
            hyperparam = numpy.log10(numpy.abs(scale))
        else:
            hyperparam = numpy.abs(scale)

        return hyperparam

    # ===================
    # hyperparam to scale
    # ===================

    def _hyperparam_to_scale(self, hyperparam):
        """
        Sets scale from hyperparam. If self.use_log_scale is True, hyperparam
        is the log10 of scale, hence, 10**hyperparam is set to scale. If
        self.use_log_scale is False, hyperparam is directly set to scale.
        """

        # If logscale is used, input hyperparam is log of the scale.
        if self.use_log_scale:
            scale = 10.0**hyperparam
        else:
            scale = numpy.abs(hyperparam)

        return scale

    # ============================
    # hyperparam to log hyperparam
    # ============================

    def hyperparam_to_log_hyperparam(self, hyperparam):
        """
        Converts the input hyperparameters to their log10, if this is enabled
        by ``self.use_scale``.

        If is assumed that the input hyperparam is not in log scale, and it
        contains either of the following form:

        * [sigma, sigma0]
        * [sigma, sigma0, scale1, scale2, ...]
        """

        if numpy.isscalar(hyperparam):
            hyperparam = numpy.array([hyperparam], dtype=float)
        elif isinstance(hyperparam, list):
            hyperparam = numpy.array(hyperparam, dtype=float)

        # Convert scale to log10 of scale
        if hyperparam.size > self.scale_index:
            if self.use_log_scale:
                scale = hyperparam[self.scale_index:]
                hyperparam[self.scale_index:] = \
                    self._scale_to_hyperparam(scale)

        return hyperparam

    # ==================
    # extract hyperparam
    # ==================

    def extract_hyperparam(self, hyperparam):
        """
        It is assumed the input hyperparam might be in the log10 scale, and
        may or may not contain scales. The output will be converted to non-log
        format and will include scale, regardless if the input has scale or
        not.
        """

        sigma, sigma0 = self._hyperparam_to_sigmas(hyperparam)
        eta = (sigma0/sigma)**2

        if hyperparam.size > self.scale_index:
            scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
        else:
            scale = self.cov.get_scale()

        return sigma, sigma0, eta, scale

    # =====
    # M dot
    # =====

    def M_dot(self, Binv, Y, sigma, sigma0, z):
        """
        Multiplies the matrix :math:`\\mathbf{M}` by a given vector
        :math:`\\boldsymbol{z}`. The matrix :math:`\\mathbf{M}` is defined by

        .. math::

            \\mathbf{M} = \\boldsymbol{\\Sigma}^{-1} \\mathbf{P},

        where the covariance matrix :math:`\\boldsymbol{\\Sigmna}` is defined
        by

        .. math::

            \\boldsymbol{\\Sigma} = \\sigma^2 \\mathbf{K} +
            \\sigma_0^2 \\mathbf{I},

        and the projection matrix :math:`\\mathbf{P}` is defined by

        .. math::

            \\mathbf{P} = \\mathbf{I} - \\mathbf{X} (\\mathbf{X}^{\\intercal}
            \\boldsymbol{\\Sigma}^{-1}) \\mathbf{X})^{-1}
            \\mathbf{X}^{\\intercal} \\boldsymbol{\\Sigma}^{-1}.

        :param cov: An object of class :class:`Covariance` which represents
            the operator :math:`\\sigma^2 \\mathbf{K} +
            \\sigma_0^2 \\mathbf{I}`.
        :type cov: gaussian_proc.Covariance

        :param Binv: The inverse of matrix
            :math:`\\mathbf{B} = \\mathbf{X}^{\\intercal} \\mathbf{Y}`.
        :type Binv: numpy.ndarray

        :param Y: The matrix
            :math:`\\mathbf{Y} = \\boldsymbol{\\Sigma}^{-1} \\mathbf{X}`.
        :type Y: numpy.ndarray

        :param sigma: The parameter :math:`\\sigma`.
        :type sigma: float

        :param sigma0: The parameter :math:`\\sigma_0`.
        :type sigma0: float

        :param z: The data column vector.
        :type z: numpy.ndarray
        """

        # Computing w = Sinv*z, where S is sigma**2 * K + sigma0**2 * I
        w = self.cov.solve(sigma, sigma0, z)

        # Computing Mz
        Ytz = numpy.matmul(Y.T, z)
        Binv_Ytz = numpy.matmul(Binv, Ytz)
        Y_Binv_Ytz = numpy.matmul(Y, Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        return Mz

    # =============
    # update Y B Mz
    # =============

    def _update_Y_B_Mz(self, hyperparam):
        """
        Computes Y, B, Binv, and Mz. These variables are shared among many of
        the functions, hence their values are stored as the class attribute to
        avoid re-computation when the hyperparam is the same.
        """

        # Check if likelihood is already computed for an identical hyperparam
        if (self.Y is None) or \
                (self.B is None) or \
                (self.Binv is None) or \
                (self.Mz is None) or \
                (self.Y_B_Mz_hyperparam is None) or \
                (hyperparam.size != self.Y_B_Mz_hyperparam.size) or \
                (not numpy.allclose(hyperparam, self.Y_B_Mz_hyperparam,
                                    atol=self.hyperparam_tol)):

            # hyperparameters
            sigma, sigma0 = self._hyperparam_to_sigmas(hyperparam)

            # Include derivative w.r.t scale
            if hyperparam.size > self.scale_index:
                scale = self._hyperparam_to_scale(
                        hyperparam[self.scale_index:])
                self.cov.set_scale(scale)

            # Variables to compute/update
            self.Y = self.cov.solve(sigma, sigma0, self.X)
            self.B = numpy.matmul(self.X.T, self.Y)
            self.Binv = numpy.linalg.inv(self.B)
            self.Mz = self.M_dot(self.Binv, self.Y, sigma, sigma0, self.z)

            # Update the current hyperparam
            self.Y_B_Mz_hyperparam = hyperparam

    # ===================
    # update MMz KMz MKMz
    # ===================

    def _update_MMz_KMz_MKMz(self, hyperparam):
        """
        Computes MMz, KMz, and MKMz.
        """

        # Check if likelihood is already computed for an identical hyperparam
        if (self.MMz is None) or \
                (self.KMz is None) or \
                (self.MKMz is None) or \
                (self.MMz_KMz_MKMz_hyperparam is None) or \
                (hyperparam.size != self.MMz_KMz_MKMz_hyperparam.size) or \
                (not numpy.allclose(hyperparam, self.MMz_KMz_MKMz_hyperparam,
                                    atol=self.hyperparam_tol)):

            # hyperparameters
            sigma, sigma0 = self._hyperparam_to_sigmas(hyperparam)

            # Set scale of the covariance object
            if hyperparam.size > self.scale_index:
                scale = self._hyperparam_to_scale(
                        hyperparam[self.scale_index:])
                self.cov.set_scale(scale)

            # Variables to update
            self.MMz = self.M_dot(self.Binv, self.Y, sigma, sigma0, self.Mz)
            self.KMz = self.cov.dot(1.0, 0.0, self.Mz)
            self.MKMz = self.M_dot(self.Binv, self.Y, sigma, sigma0, self.KMz)

            # Update the current hyperparam
            self.MMz_KMz_MKMz_hyperparam = hyperparam

    # ==============
    # update trace M
    # ==============

    def _update_trace_M(self, hyperparam):
        """
        Computes trace of M.
        """

        # Check if likelihood is already computed for an identical hyperparam
        if (self.trace_M is None) or \
                (self.trace_M_hyperparam is None) or \
                (hyperparam.size != self.trace_M_hyperparam.size) or \
                (not numpy.allclose(hyperparam, self.trace_M_hyperparam,
                                    atol=self.hyperparam_tol)):

            # hyperparameters
            sigma, sigma0 = self._hyperparam_to_sigmas(hyperparam)

            # Set scale of the covariance object
            if hyperparam.size > self.scale_index:
                scale = self._hyperparam_to_scale(
                        hyperparam[self.scale_index:])
                self.cov.set_scale(scale)

            # Compute trace of M
            if numpy.abs(sigma) < self.cov.tol:
                n, m = self.X.shape
                self.trace_M = (n - m) / sigma0**2
            else:
                trace_Sinv = self.cov.traceinv(sigma, sigma0)
                YtY = numpy.matmul(self.Y.T, self.Y)
                trace_BinvYtY = numpy.trace(numpy.matmul(self.Binv, YtY))
                self.trace_M = trace_Sinv - trace_BinvYtY

            # Update the current hyperparam
            self.trace_M_hyperparam = hyperparam

    # =========================
    # update Sinv KpSinv SpSinv
    # =========================

    def _update_Sinv_KpSinv_SpSinv(self, hyperparam):
        """
        Compute Sinv, SpSinv.
        """

        # Check if likelihood is already computed for an identical hyperparam
        if (self.Sinv is None) or \
                (self.KpSinv is None) or \
                (self.SpSinv is None) or \
                (self.Sinv_KpSinv_SpSinv_hyperparam is None) or \
                (hyperparam.size !=
                    self.Sinv_KpSinv_SpSinv_hyperparam.size) or \
                (not numpy.allclose(hyperparam,
                                    self.Sinv_KpSinv_SpSinv_hyperparam,
                                    atol=self.hyperparam_tol)):

            # hyperparameters
            sigma, sigma0 = self._hyperparam_to_sigmas(hyperparam)

            # Set scale of the covariance object
            scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
            self.cov.set_scale(scale)

            # Update variables
            S = self.cov.get_matrix(sigma, sigma0)
            self.Sinv = numpy.linalg.inv(S)

            # Initialize KpSinv and SpSinv as list of size of scale.size
            self.KpSinv = [None] * scale.size
            self.SpSinv = [None] * scale.size

            for p in range(scale.size):
                # Kp = self.cov.get_matrix(1.0, 0.0, derivative=[p])
                # self.KpSinv[p] = Kp @ self.Sinv
                # self.SpSinv[p] = sigma**2 * self.KpSinv[p]

                Sp = self.cov.get_matrix(sigma, sigma0, derivative=[p])
                self.SpSinv[p] = Sp @ self.Sinv
                self.KpSinv[p] = self.SpSinv[p] / sigma**2

            # Update the current hyperparam
            self.Sinv_KpSinv_SpSinv_hyperparam = hyperparam

    # ==========
    # likelihood
    # ==========

    def likelihood(self, sign_switch, hyperparam):
        """
        Returns the log-likelihood function.

        Hyperparam are in one of the two forms:
        * [sigma, sigma0]
        * [sigma, sigma0, scale0, scale1, ...]

        ``sign_switch`` changes the sign of the output from ``ell`` to
        ``-ell``. When ``True``, this is used to minimizing (instead of
        maximizing) the negative of log-likelihood function.
        """

        # Check if likelihood is already computed for an identical hyperparam
        if (self.ell_hyperparam is not None) and \
                (self.ell is not None) and \
                (hyperparam.size == self.ell_hyperparam.size) and \
                numpy.allclose(hyperparam, self.ell_hyperparam,
                               atol=self.hyperparam_tol):
            if sign_switch:
                return -self.ell
            else:
                return self.ell

        # hyperparameters
        sigma, sigma0 = self._hyperparam_to_sigmas(hyperparam)

        # Include derivative w.r.t scale
        if hyperparam.size > self.scale_index:
            scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
            self.cov.set_scale(scale)

        n, m = self.X.shape

        # cov is the (sigma**2) * K + (sigma0**2) * I
        logdet_S = self.cov.logdet(sigma, sigma0)

        # Compute (or update) Y, B, Mz
        self._update_Y_B_Mz(hyperparam)

        # Compute zMz
        zMz = numpy.dot(self.z, self.Mz)

        # Compute log det (X.T*Sinv*X)
        logdet_B = numpy.log(numpy.linalg.det(self.B))

        # Log likelihood
        ell = -0.5*(n-m)*numpy.log(2.0*numpy.pi) - 0.5*logdet_S \
            - 0.5*logdet_B - 0.5*zMz

        # Store ell to member data (without sign-switch).
        self.ell = ell
        self.ell_hyperparam = hyperparam

        # If ell is used in scipy.optimize.minimize, change the sign to obtain
        # the minimum of -ell
        if sign_switch:
            ell = -ell

        return ell

    # ======================
    # likelihood der1 sigmas
    # ======================

    def _likelihood_der1_sigmas(self, hyperparam):
        """
        Computes the first derivative of log-likelihood w.r.t sigma and sigma0.
        """

        # hyperparameters
        sigma, sigma0 = self._hyperparam_to_sigmas(hyperparam)

        # Set scale of the covariance object
        if hyperparam.size > self.scale_index:
            scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
            self.cov.set_scale(scale)

        n, m = self.X.shape

        # Compute (or update) Y, Binv, Mz
        self._update_Y_B_Mz(hyperparam)

        # Compute KMz (Setting sigma=1 and sigma0=0 to have cov = K)
        KMz = self.cov.dot(1.0, 0.0, self.Mz)

        # Compute zMMz and zMKMz
        zMMz = numpy.dot(self.Mz, self.Mz)
        zMKMz = numpy.dot(self.Mz, KMz)

        # Compute (or update) trace of M
        self._update_trace_M(hyperparam)

        # Compute trace of KM which is (n-m)/sigma**2 - eta* trace(M)
        if numpy.abs(sigma) < self.cov.tol:
            YtKY = numpy.matmul(self.Y.T, self.cov.dot(1.0, 0.0, self.Y))
            BinvYtKY = numpy.matmul(self.Binv, YtKY)
            trace_BinvYtKY = numpy.trace(BinvYtKY)
            trace_KM = n/sigma0**2 - trace_BinvYtKY
        else:
            eta = (sigma0 / sigma)**2
            trace_KM = (n - m)/sigma**2 - eta*self.trace_M

        # Derivative of ell w.r.t to sigma
        dell_dsigma = -0.5*trace_KM + 0.5*zMKMz
        dell_dsigma0 = -0.5*self.trace_M + 0.5*zMMz

        # Concatenate derivative w.r.t to both sigma and sigma0 (Jacobian)
        jacobian = numpy.array([dell_dsigma, dell_dsigma0])

        return jacobian

    # ======================
    # likelihood der2 sigmas
    # ======================

    def _likelihood_der2_sigmas(self, hyperparam):
        """
        Computes the second derivatives of log-likelihood w.r.t sigma and
        sigma0, and the mixed derivative w.r.t sigma and sigma0.
        """

        # hyperparameters
        sigma, sigma0 = self._hyperparam_to_sigmas(hyperparam)
        eta = (sigma0 / sigma)**2

        # Include derivative w.r.t scale
        if hyperparam.size > self.scale_index:
            scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
            self.cov.set_scale(scale)

        n, m = self.X.shape

        # Compute (or update) Y, Binv, Mz
        self._update_Y_B_Mz(hyperparam)

        # Computing Y=Sinv*X, V = Sinv*Y, and w=Sinv*z
        V = self.cov.solve(sigma, sigma0, self.Y)

        # B is Xt * Y
        YtY = numpy.matmul(self.Y.T, self.Y)
        A = numpy.matmul(self.Binv, YtY)

        # Compute MMz, KMz, MKMz
        self._update_MMz_KMz_MKMz(hyperparam)

        # Compute KMz, zMMMz (Setting sigma=1 and sigma0=0 to have cov=K)
        zMMMz = numpy.dot(self.Mz, self.MMz)

        # Compute zMKMKMz
        zMMKMz = numpy.dot(self.MMz, self.KMz)
        zMKMKMz = numpy.dot(self.KMz, self.MKMz)

        # Compute (or update) trace of M
        self._update_trace_M(hyperparam)

        # Trace of Sinv**2
        trace_S2inv = self.cov.traceinv(sigma, sigma0, exponent=2)

        # Trace of M**2
        YtV = numpy.matmul(self.Y.T, V)
        F = numpy.matmul(self.Binv, YtV)
        trace_F = numpy.trace(F)
        AA = numpy.matmul(A, A)
        trace_AA = numpy.trace(AA)
        trace_M2 = trace_S2inv - 2.0*trace_F + trace_AA

        # Trace of (KM)**2
        if numpy.abs(sigma) < self.cov.tol:
            trace_K2 = self.cov.trace(1.0, 0.0, exponent=2)
            D = numpy.matmul(self.X.T, self.X)
            Dinv = numpy.linalg.inv(D)
            KX = self.cov.dot(1.0, 0.0, self.X, exponent=1)
            XKX = numpy.matmul(self.X.T, KX)
            XK2X = numpy.matmul(KX.T, KX)
            E = numpy.matmul(Dinv, XKX)
            E2 = numpy.matmul(E, E)
            F = numpy.matmul(Dinv, XK2X)
            trace_KMKM = (trace_K2 - 2.0*numpy.trace(F) + numpy.trace(E2)) / \
                sigma0**4
        else:
            trace_KMKM = (n-m)/sigma**4 - (2*eta/sigma**2)*self.trace_M + \
                (eta**2)*trace_M2

        # Trace of K*(M**2)
        if numpy.abs(sigma) < self.cov.tol:
            trace_KM = (n - numpy.trace(E))/sigma0**2
            trace_KMM = trace_KM / sigma0**2
        else:
            trace_KMM = self.trace_M/sigma**2 - eta*trace_M2

        # Compute second derivatives
        d2ell_dsigma0_dsigma0 = 0.5*trace_M2 - zMMMz
        d2ell_dsigma_dsigma = 0.5*trace_KMKM - zMKMKMz
        d2ell_dsigma_dsigma0 = 0.5*trace_KMM - zMMKMz

        # Concatenate all second derivatives to form Hessian
        hessian = numpy.array(
                [[d2ell_dsigma_dsigma, d2ell_dsigma_dsigma0],
                 [d2ell_dsigma_dsigma0, d2ell_dsigma0_dsigma0]], dtype=float)

        return hessian

    # =====================
    # likelihood der1 scale
    # =====================

    def _likelihood_der1_scale(self, hyperparam):
        """
        Computes the first derivative of log-likelihood w.r.t scale.
        """

        # hyperparameters
        sigma, sigma0 = self._hyperparam_to_sigmas(hyperparam)

        # Set scale of the covariance object
        scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
        self.cov.set_scale(scale)

        dell_dscale = numpy.zeros((scale.size, ), dtype=float)

        # Compute (or update) Y, Binv, Mz
        self._update_Y_B_Mz(hyperparam)

        # Compute Sinv, SpSinv
        if not self.stochastic_traceinv:
            self._update_Sinv_KpSinv_SpSinv(hyperparam)

        # Sp is the derivative of cov w.r.t the p-th element of scale.
        for p in range(scale.size):

            # Compute zMSpMz
            SpMz = self.cov.dot(sigma, sigma0, self.Mz, derivative=[p])
            zMSpMz = numpy.dot(self.Mz, SpMz)

            # Compute the first component of trace of Sp * M
            if self.stochastic_traceinv:
                # Compute traceinv using stochastic estimation method. Note
                # that since Sp is not positive-definite, we cannot use
                # Cholesky method in imate. The only viable option is
                # Hutchinson's method.
                Sp = self.cov.get_matrix(sigma, sigma0, derivative=[p])
                trace_SpSinv = self.cov.traceinv(sigma, sigma0, B=Sp,
                                                 imate_method='hutchinson')
            else:
                # Using exact method (compute inverse directly)
                trace_SpSinv, _ = imate.trace(self.SpSinv[p], method='exact')

            # Compute the second component of trace of Sp * M
            SpY = self.cov.dot(sigma, sigma0, self.Y, derivative=[p])
            YtSpY = numpy.matmul(self.Y.T, SpY)
            BinvYtSpY = numpy.matmul(self.Binv, YtSpY)
            trace_BinvYtSpY = numpy.trace(BinvYtSpY)

            # Compute trace of Sp * M
            trace_SpM = trace_SpSinv - trace_BinvYtSpY

            # Derivative of ell w.r.t p-th element of distance scale
            dell_dscale[p] = -0.5*trace_SpM + 0.5*zMSpMz

        return dell_dscale

    # =====================
    # likelihood der2 mixed
    # =====================

    def _likelihood_der2_mixed(self, hyperparam):
        """
        Computes the second mixed derivative of log-likelihood w.r.t scale and
        sigma, also w.r.t scale and sigma0.
        """

        # hyperparameters
        sigma, sigma0 = self._hyperparam_to_sigmas(hyperparam)

        # Set scale of the covariance object
        scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
        self.cov.set_scale(scale)

        # Initialize arrays
        d2ell_dsigma_dscale = numpy.zeros((scale.size), dtype=float)
        d2ell_dsigma0_dscale = numpy.zeros((scale.size), dtype=float)
        d2ell_mixed = numpy.zeros((2, scale.size), dtype=float)

        # Compute (or update) Y, Binv, Mz
        self._update_Y_B_Mz(hyperparam)

        YtY = numpy.matmul(self.Y.T, self.Y)
        A = numpy.matmul(self.Binv, YtY)

        # Compute MMz, KMz, MKMz
        self._update_MMz_KMz_MKMz(hyperparam)

        # Common variables
        K = self.cov.get_matrix(1.0, 0.0, derivative=[])
        KY = self.cov.dot(1.0, 0.0, self.Y)
        YtKY = numpy.matmul(self.Y.T, KY)
        Dk = numpy.matmul(self.Binv, YtKY)

        # Compute Sinv, KpSinv, SpSinv
        if not self.stochastic_traceinv:
            self._update_Sinv_KpSinv_SpSinv(hyperparam)
            KSinv = numpy.matmul(K, self.Sinv)

        # Sp is the derivative of cov w.r.t the p-th element of scale. Spq
        # is the second mixed derivative of S w.r.t p-th and q-th elements
        # of scale.
        for p in range(scale.size):

            # -----------------------------------------------
            # 1. Compute mixed derivatives of scale and sigma
            # -----------------------------------------------

            # 1.1. Compute zMSpMKMz
            SpMz = self.cov.dot(sigma, sigma0, self.Mz, derivative=[p])
            zMSpMKMz = numpy.dot(SpMz, self.MKMz)

            # 1.2. Compute zMKpMz
            KpMz = self.cov.dot(1.0, 0.0, self.Mz, derivative=[p])
            zMKpMz = numpy.dot(self.Mz, KpMz)

            # 1.3. Compute trace of Kp * M

            # Compute the first component of trace of Kp * M
            if self.stochastic_traceinv:

                # Computing traceinv using either cholesky or hutchinson
                Kp = self.cov.get_matrix(1.0, 0.0, derivative=[p])
                Sp = sigma**2 * Kp

                # Note that since Kp is not positive-definite, we cannot use
                # Cholesky method in imate. The only viable option is
                # Hutchinson's method.
                trace_KpSinv = self.cov.traceinv(sigma, sigma0, B=Kp,
                                                 imate_method='hutchinson')
            else:
                trace_KpSinv, _ = imate.trace(self.KpSinv[p], method='exact')

            # Compute the second component of trace of Kp * M
            KpY = self.cov.dot(1.0, 0.0, self.Y, derivative=[p])
            YtKpY = numpy.matmul(self.Y.T, KpY)
            BinvYtKpY = numpy.matmul(self.Binv, YtKpY)
            trace_BinvYtKpY = numpy.trace(BinvYtKpY)

            # Compute trace of Kp * M
            trace_KpM = trace_KpSinv - trace_BinvYtKpY

            # 1.4. Compute trace of K * M * Sp * M

            # Compute first part of trace of K * M * Sp * M
            if self.stochastic_traceinv:
                # Compute traceinv with stochastic estimation method
                trace_KMSpM_1 = self.cov.traceinv(sigma, sigma0, B=Sp, C=K,
                                                  imate_method='hutchinson')
            else:
                # Compute traceinv using direct computation of inverse matrix
                KSinvSpSinv = numpy.matmul(KSinv, self.SpSinv[p])
                trace_KMSpM_1, _ = imate.trace(KSinvSpSinv, method='exact')

            # Compute the second part of trace of K * M * Sp * M
            SpY = self.cov.dot(sigma, sigma0, self.Y, derivative=[p])
            SinvSpY = self.cov.solve(sigma, sigma0, SpY)
            YtKSinvSpY = numpy.matmul(KY.T, SinvSpY)
            F21 = numpy.matmul(self.Binv, YtKSinvSpY)
            F22 = numpy.matmul(self.Binv, YtKSinvSpY.T)
            trace_KMSpM_21 = numpy.trace(F21)
            trace_KMSpM_22 = numpy.trace(F22)

            # Compute the third part of trace of K * M * Sp * M
            YtSpY = numpy.matmul(self.Y.T, SpY)
            Dp = numpy.matmul(self.Binv, YtSpY)
            DkDp = numpy.matmul(Dk, Dp)
            trace_KMSpM_3 = numpy.trace(DkDp)

            # Compute trace of K * M * Sp * M
            trace_KMSpM = trace_KMSpM_1 - trace_KMSpM_21 - \
                trace_KMSpM_22 + trace_KMSpM_3

            # 1.5. Second derivatives w.r.t scale
            d2ell_dsigma_dscale[p] = -0.5*trace_KpM + 0.5*trace_KMSpM - \
                zMSpMKMz + 0.5*zMKpMz

            # ------------------------------------------------
            # 2. Compute mixed derivatives of scale and sigma0
            # ------------------------------------------------

            # 2.1. Compute zMSpMMz
            zMSpMMz = numpy.dot(SpMz, self.MMz)

            # 2.4. Compute trace of M * Sp * M

            # Compute first part of trace of M * Sp * M
            if self.stochastic_traceinv:
                # Compute traceinv using stochastic estimation method. Note
                # that since Sp is not positive-definite, we cannot use
                # Cholesky method in imate. The only viable option is
                # Hutchinson's method.
                trace_MSpM_1 = self.cov.traceinv(sigma, sigma0, B=Sp,
                                                 exponent=2,
                                                 imate_method='hutchinson')
            else:
                SinvSpSinv = numpy.matmul(self.Sinv, self.SpSinv[p])
                trace_MSpM_1, _ = imate.trace(SinvSpSinv, method='exact')

            # Compute the second part of trace of M * Sp * M
            YtSinvSpY = numpy.matmul(self.Y.T, SinvSpY)
            G21 = numpy.matmul(self.Binv, YtSinvSpY)
            G22 = numpy.matmul(self.Binv, YtSinvSpY.T)
            trace_MSpM_21 = numpy.trace(G21)
            trace_MSpM_22 = numpy.trace(G22)

            # Compute the third part of trace of M * Sp * M
            DpA = numpy.matmul(Dp, A)
            trace_MSpM_3 = numpy.trace(DpA)

            # Compute trace of M * Sp * M
            trace_MSpM = trace_MSpM_1 - trace_MSpM_21 - trace_MSpM_22 + \
                trace_MSpM_3

            # 2.5. Second derivatives w.r.t scale
            d2ell_dsigma0_dscale[p] = 0.5*trace_MSpM - zMSpMMz

            # Concatenate mixed derivatives of scale and sigmas
            d2ell_mixed[0, :] = d2ell_dsigma_dscale
            d2ell_mixed[1, :] = d2ell_dsigma0_dscale

        return d2ell_mixed

    # =====================
    # likelihood der2 scale
    # =====================

    def _likelihood_der2_scale(self, hyperparam):
        """
        Computes the second derivative of log-likelihood w.r.t scale.
        """

        # hyperparameters
        sigma, sigma0 = self._hyperparam_to_sigmas(hyperparam)

        # Set scale of the covariance object
        scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
        self.cov.set_scale(scale)

        # Initialize output array
        d2ell_dscale2 = numpy.zeros((scale.size, scale.size), dtype=float)

        # Compute (or update) Y, Binv, Mz
        self._update_Y_B_Mz(hyperparam)

        # Compute Sinv, SpSinv
        if not self.stochastic_traceinv:
            self._update_Sinv_KpSinv_SpSinv(hyperparam)

        for p in range(scale.size):

            Sp = self.cov.get_matrix(sigma, sigma0, derivative=[p])
            SpMz = self.cov.dot(sigma, sigma0, self.Mz, derivative=[p])
            MSpMz = self.M_dot(self.Binv, self.Y, sigma, sigma0, SpMz)

            for q in range(p, scale.size):

                # 1. Compute zMSqMSpMz
                if p == q:
                    SqMz = SpMz
                else:
                    SqMz = self.cov.dot(sigma, sigma0, self.Mz, derivative=[q])
                zMSqMSpMz = numpy.dot(SqMz, MSpMz)

                # 2. Compute zMSpqMz
                SpqMz = self.cov.dot(sigma, sigma0, self.Mz, derivative=[p, q])
                zMSpqMz = numpy.dot(self.Mz, SpqMz)

                # 3. Computing trace of Spq * M in three steps

                # Compute the first component of trace of Spq * M
                Spq = self.cov.get_matrix(sigma, sigma0, derivative=[p, q])
                if self.stochastic_traceinv:
                    # Compute traceinv using stochastic estimation method. Note
                    # that since Spq is not positive-definite, we cannot use
                    # Cholesky method in imate. The only viable option is
                    # Hutchinson's method.
                    trace_SpqSinv = self.cov.traceinv(
                            sigma, sigma0, B=Spq, imate_method='hutchinson')
                else:
                    SpqSinv = numpy.matmul(Spq, self.Sinv)
                    trace_SpqSinv, _ = imate.trace(SpqSinv, method='exact')

                # Compute the second component of trace of Spq * M
                SpqY = self.cov.dot(sigma, sigma0, self.Y, derivative=[p, q])
                YtSpqY = numpy.matmul(self.Y.T, SpqY)
                BinvYtSpqY = numpy.matmul(self.Binv, YtSpqY)
                trace_BinvYtSpqY = numpy.trace(BinvYtSpqY)

                # Compute trace of Spq * M
                trace_SpqM = trace_SpqSinv - trace_BinvYtSpqY

                # 4. Compute trace of Sp * M * Sq * M

                # Compute first part of trace of Sp * M * Sq * M
                Sq = self.cov.get_matrix(sigma, sigma0, derivative=[q])
                if self.stochastic_traceinv:
                    trace_SpMSqM_1 = self.cov.traceinv(
                            sigma, sigma0, B=Sq, C=Sp,
                            imate_method='hutchinson')
                else:
                    SpSinvSqSinv = numpy.matmul(self.SpSinv[p], self.SpSinv[q])
                    trace_SpMSqM_1, _ = imate.trace(SpSinvSqSinv,
                                                    method='exact')

                # Compute the second part of trace of Sp * M * Sq * M
                SpY = numpy.matmul(Sp, self.Y)
                if p == q:
                    SqY = SpY
                else:
                    SqY = numpy.matmul(Sq, self.Y)
                SinvSqY = self.cov.solve(sigma, sigma0, SqY)
                YtSpSinvSqY = numpy.matmul(SpY.T, SinvSqY)
                F21 = numpy.matmul(self.Binv, YtSpSinvSqY)
                F22 = numpy.matmul(self.Binv, YtSpSinvSqY.T)
                trace_SpMSqM_21 = numpy.trace(F21)
                trace_SpMSqM_22 = numpy.trace(F22)

                # Compute the third part of trace of Sp * M * Sq * M
                YtSpY = numpy.matmul(self.Y.T, SpY)
                if p == q:
                    YtSqY = YtSpY
                else:
                    YtSqY = numpy.matmul(self.Y.T, SqY)
                Dp = numpy.matmul(self.Binv, YtSpY)
                if p == q:
                    Dq = Dp
                else:
                    Dq = numpy.matmul(self.Binv, YtSqY)
                DpDq = numpy.matmul(Dp, Dq)
                trace_SpMSqM_3 = numpy.trace(DpDq)

                # Compute trace of Sp * M * Sq * M
                trace_SpMSqM = trace_SpMSqM_1 - trace_SpMSqM_21 - \
                    trace_SpMSqM_22 + trace_SpMSqM_3

                # 5. Second derivatives w.r.t scale
                d2ell_dscale2[p, q] = -0.5*trace_SpqM + \
                    0.5*trace_SpMSqM - zMSqMSpMz + 0.5*zMSpqMz

                if p != q:
                    d2ell_dscale2[q, p] = d2ell_dscale2[p, q]

        return d2ell_dscale2

    # ===================
    # likelihood jacobian
    # ===================

    def likelihood_jacobian(self, sign_switch, hyperparam):
        """
        When both :math:`\\sigma` and :math:`\\sigma_0` are zero, Jacobian is
        undefined.
        """

        # Check if Jacobian is already computed for an identical hyperparam
        if (self.ell_jacobian_hyperparam is not None) and \
                (self.ell_jacobian is not None) and \
                (hyperparam.size == self.ell_jacobian_hyperparam.size) and \
                numpy.allclose(hyperparam, self.ell_jacobian_hyperparam,
                               atol=self.hyperparam_tol):
            if sign_switch:
                return -self.ell_jacobian
            else:
                return self.ell_jacobian

        # Compute first derivative w.r.t sigma and sigma0
        jacobian = self._likelihood_der1_sigmas(hyperparam)

        # Compute Jacobian w.r.t scale
        if hyperparam.size > self.scale_index:

            # Compute first derivative w.r.t scale
            dell_dscale = self._likelihood_der1_scale(hyperparam)

            # Convert derivative w.r.t log of scale
            if self.use_log_scale:
                scale = self._hyperparam_to_scale(
                        hyperparam[self.scale_index:])
                for p in range(scale.size):
                    dell_dscale[p] = dell_dscale[p] * scale[p] * \
                        numpy.log(10.0)

            # Concatenate jacobian
            jacobian = numpy.r_[jacobian, dell_dscale]

        # Store jacobian to member data (without sign-switch).
        self.ell_jacobian = jacobian
        self.ell_jacobian_hyperparam = hyperparam

        if sign_switch:
            jacobian = -jacobian

        return jacobian

    # ==================
    # likelihood hessian
    # ==================

    def likelihood_hessian(self, sign_switch, hyperparam):
        """
        """

        # Check if Hessian is already computed for an identical hyperparam
        if (self.ell_hessian_hyperparam is not None) and \
                (self.ell_hessian is not None) and \
                (hyperparam.size == self.ell_hessian_hyperparam.size) and \
                numpy.allclose(hyperparam, self.ell_hessian_hyperparam,
                               atol=self.hyperparam_tol):
            if sign_switch:
                return -self.ell_hessian
            else:
                return self.ell_hessian

        # Second derivatives w.r.t sigma and sigma0 and their mixed derivative
        hessian = self._likelihood_der2_sigmas(hyperparam)

        # Compute Hessian w.r.t scale
        if hyperparam.size > self.scale_index:

            # Compute second derivative w.r.t scale
            d2ell_dscale2 = self._likelihood_der2_scale(hyperparam)

            # Compute second mixed derivative of scale w.r.t sigma and sigma0
            d2ell_mixed = self._likelihood_der2_mixed(hyperparam)

            # Convert derivative w.r.t log of scale (if needed)
            if self.use_log_scale:

                scale = self._hyperparam_to_scale(
                        hyperparam[self.scale_index:])

                for p in range(scale.size):
                    # Mixed derivative of scale and sigma
                    d2ell_mixed[0, p] = d2ell_mixed[0, p] * \
                        scale[p] * numpy.log(10.0)

                    # Mixed derivative of eta and sigma0
                    d2ell_mixed[1, p] = d2ell_mixed[1, p] * \
                        scale[p] * numpy.log(10.0)

                # To convert derivative to log scale, Jacobian is needed. Note:
                # The Jacobian itself is already converted to log scale.
                jacobian_ = self.likelihood_jacobian(False, hyperparam)

                # Second derivative w.r.t scale
                dell_dscale = jacobian_[self.scale_index:]

                for p in range(scale.size):
                    for q in range(scale.size):
                        if p == q:

                            # dell_dscale is already converted to logscale
                            d2ell_dscale2[p, q] = d2ell_dscale2[p, q] * \
                                scale[p]**2 * (numpy.log(10.0)**2) + \
                                dell_dscale[p] * numpy.log(10.0)
                        else:
                            d2ell_dscale2[p, q] = d2ell_dscale2[p, q] * \
                                scale[p] * scale[q] * (numpy.log(10.0)**2)

            # Concatenate all mixed derivatives
            hessian = numpy.block([[hessian, d2ell_mixed],
                                   [d2ell_mixed.T, d2ell_dscale2]])

        # Store hessian to member data (without sign-switch).
        self.ell_hessian = hessian
        self.ell_hessian_hyperparam = hyperparam

        if sign_switch:
            hessian = -hessian

        return hessian
