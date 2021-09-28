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

import time
import numpy
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.optimize
from functools import partial

from .._utilities.plot_utilities import *                    # noqa: F401, F403
from .._utilities.plot_utilities import load_plot_settings, save_plot, plt, \
        mark_inset, InsetPosition, matplotlib, make_axes_locatable
from ._root_finding import find_interval_with_sign_change, chandrupatla_method
import imate
import warnings


# ==================
# Profile Likelihood
# ==================

class ProfileLikelihood(object):
    """
    Likelihood function that is profiled with respect to :math:`\\sigma`
    variable.
    """

    # ====
    # init
    # ====

    def __init__(self, z, X, cov):
        """
        Initialization.
        """

        # Attributes
        self.z = z
        self.X = X
        self.mixed_cor = cov.mixed_cor

        # Configuration
        self.hyperparam_tol = 1e-8
        self.use_log_eta = True
        self.use_log_scale = True

        # Store ell, its Jacobian and Hessian.
        self.ell = None
        self.ell_jacobian = None
        self.ell_hessian = None

        # Store hyperparam used to compute ell, its Jacobian and Hessian.
        self.ell_hyperparam = None
        self.ell_jacobian_hyperparam = None
        self.ell_hessian_hyperparam = None

        # Internal settings
        self.max_eta = 1e+16
        self.min_eta = 1e-16

    # =================
    # eta to hyperparam
    # =================

    def _eta_to_hyperparam(self, eta):
        """
        Sets hyperparam from eta. eta is always given as natural scale (i.e.,
        not as in log scale). If self.use_log_eta is True, hyperparam is set
        as log10 of eta, otherwise, just as eta.
        """

        # If logscale is used, output hyperparam is log of eta.
        if self.use_log_eta:
            hyperparam = numpy.log10(numpy.abs(eta))
        else:
            hyperparam = numpy.abs(eta)

        return hyperparam

    # =================
    # hyperparam to eta
    # =================

    def _hyperparam_to_eta(self, hyperparam):
        """
        Sets eta from hyperparam. If self.use_log_eta is True, hyperparam is
        the log10 of eta, hence, 10**hyperparam is set to eta. If
        self.use_log_eta is False, hyperparam is directly set to eta.
        """

        # Using only the first component (if an array is given)
        if numpy.isscalar(hyperparam):
            hyperparam_ = hyperparam
        else:
            hyperparam_ = hyperparam[0]

        # If logscale is used, input hyperparam is log of eta.
        if self.use_log_eta:
            eta = 10.0**hyperparam_
        else:
            eta = numpy.abs(hyperparam_)

        return eta

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
        Sets scale from hyperparam. If self.use_log_eta is True, hyperparam is
        the log10 of scale, hence, 10**hyperparam is set to scale. If
        self.use_log_eta is False, hyperparam is directly set to scale.
        """

        # If logscale is used, input hyperparam is log of the scale.
        if self.use_log_scale:
            scale = 10.0**hyperparam
        else:
            scale = numpy.abs(hyperparam)

        return scale

    # =====
    # M dot
    # =====

    def M_dot(self, Binv, Y, eta, z):
        """
        Multiplies the matrix :math:`\\mathbf{M}` by a given vector
        :math:`\\boldsymbol{z}`. The matrix :math:`\\mathbf{M}` is defined by

        .. math::

            \\mathbf{M} = \\boldsymbol{\\Sigma}^{-1} \\mathbf{P},

        where the covarance matrix :math:`\\boldsymbol{\\Sigmna}` is defined by

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
        w = self.mixed_cor.solve(eta, z)

        # Computing Mz
        Ytz = numpy.matmul(Y.T, z)
        Binv_Ytz = numpy.matmul(Binv, Ytz)
        Y_Binv_Ytz = numpy.matmul(Y, Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        return Mz

    # ==========
    # Likelihood
    # ==========

    def likelihood(self, sign_switch, hyperparam):
        """
        Log likelihood function

            L = -(1/2) log det(S) - (1/2) log det(X.T*Sinv*X) -
                (1/2) sigma^(-2) * z.T * M1 * z

        where
            S = sigma^2 Kn is the covariance
            Sinv is the inverse of S
            M1 = Sinv = Sinv*X*(X.T*Sinv*X)^(-1)*X.T*Sinv

        hyperparam = [eta, scale[0], scale[1], ...]

        sign_switch changes the sign of the output from ell to -ell. When True,
        this is used to minimizing (instead of maximizing) the negative of
        log-likelihood function.
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

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        # Extract scale from hyperparam
        if (not numpy.isscalar(hyperparam)) and (hyperparam.size > 1):

            # Set scale of the covariance object
            scale = self._hyperparam_to_scale(hyperparam[1:])
            self.mixed_cor.set_scale(scale)

        n, m = self.X.shape

        if numpy.abs(eta) >= self.max_eta:

            B = numpy.matmul(self.X.T, self.X)
            Binv = numpy.linalg.inv(B)
            logdet_Binv = numpy.log(numpy.linalg.det(Binv))

            # Optimal sigma0 when eta is very large
            sigma0 = self._find_optimal_sigma0()

            # Log likelihood
            ell = -0.5*(n-m)*numpy.log(2.0*numpy.pi) \
                - (n-m)*numpy.log(sigma0) - 0.5*logdet_Binv - 0.5*(n-m)

        else:

            sigma = self._find_optimal_sigma(eta)
            logdet_Kn = self.mixed_cor.logdet(eta)

            # Compute log det (X.T Kn_inv X)
            Y = self.mixed_cor.solve(eta, self.X)

            B = numpy.matmul(self.X.T, Y)
            logdet_B = numpy.log(numpy.linalg.det(B))

            # Log likelihood
            ell = -0.5*(n-m)*numpy.log(2.0*numpy.pi) \
                - (n-m)*numpy.log(sigma) - 0.5*logdet_Kn \
                - 0.5*logdet_B \
                - 0.5*(n-m)

        # Store ell to member data (without sign-switch).
        self.ell = ell
        self.ell_hyperparam = hyperparam

        # If ell is used in scipy.optimize.minimize, change the sign to obtain
        # the minimum of -ell
        if sign_switch:
            ell = -ell

        return ell

    # ===================
    # likelihood der1 eta
    # ===================

    def _likelihood_der1_eta(self, hyperparam):
        """
        ell is the log likelihood probability. dell_deta is d(ell)/d(eta),
        which is the derivative of ell with respect to eta when the optimal
        value of sigma is substituted in the likelihood function per given eta.
        """

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        # Include derivative w.r.t scale
        if (not numpy.isscalar(hyperparam)) and (hyperparam.size > 1):

            # Set scale of the covariance object
            scale = self._hyperparam_to_scale(hyperparam[1:])
            self.mixed_cor.set_scale(scale)

        # Compute Kn_inv*X and Kn_inv*z
        Y = self.mixed_cor.solve(eta, self.X)
        w = self.mixed_cor.solve(eta, self.z)

        n, m = self.X.shape

        # Compute Mz
        B = numpy.matmul(self.X.T, Y)
        Binv = numpy.linalg.inv(B)
        Ytz = numpy.matmul(Y.T, self.z)
        Binv_Ytz = numpy.matmul(Binv, Ytz)
        Y_Binv_Ytz = numpy.matmul(Y, Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        # Traces
        trace_Kninv = self.mixed_cor.traceinv(eta)
        YtY = numpy.matmul(Y.T, Y)
        trace_BinvYtY = numpy.trace(numpy.matmul(Binv, YtY))
        trace_M = trace_Kninv - trace_BinvYtY

        # Derivative of log likelihood
        zMz = numpy.dot(self.z, Mz)
        zM2z = numpy.dot(Mz, Mz)
        sigma2 = zMz/(n-m)
        dell_deta = -0.5*(trace_M - zM2z/sigma2)

        # Return as scalar or array of length one
        if numpy.isscalar(hyperparam):
            return dell_deta
        else:
            return numpy.array([dell_deta], dtype=float)

    # ===================
    # likelihood der2 eta
    # ===================

    def _likelihood_der2_eta(self, hyperparam):
        """
        The second derivative of ell is computed as a function of only eta.
        Here, we substituted optimal value of sigma, which is self is a
        function of eta.
        """

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        # Include derivative w.r.t scale
        if (not numpy.isscalar(hyperparam)) and (hyperparam.size > 1):

            # Set scale of the covariance object
            scale = self._hyperparam_to_scale(hyperparam[1:])
            self.mixed_cor.set_scale(scale)

        Y = self.mixed_cor.solve(eta, self.X)
        V = self.mixed_cor.solve(eta, Y)
        w = self.mixed_cor.solve(eta, self.z)

        n, m = self.X.shape

        # Compute M*z
        B = numpy.matmul(self.X.T, Y)
        Binv = numpy.linalg.inv(B)
        Ytz = numpy.matmul(Y.T, self.z)
        Binv_Ytz = numpy.matmul(Binv, Ytz)
        Y_Binv_Ytz = numpy.matmul(Y, Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        # Trace of M
        # trace_Kninv = self.mixed_cor.traceinv(eta)
        YtY = numpy.matmul(Y.T, Y)
        A = numpy.matmul(Binv, YtY)
        # trace_A = numpy.trace(A)
        # trace_M = trace_Kninv - trace_A

        # Trace of M**2
        trace_Kn2inv = self.mixed_cor.traceinv(eta, exponent=2)
        YtV = numpy.matmul(Y.T, V)
        C = numpy.matmul(Binv, YtV)
        trace_C = numpy.trace(C)
        AA = numpy.matmul(A, A)
        trace_AA = numpy.trace(AA)
        trace_M2 = trace_Kn2inv - 2.0*trace_C + trace_AA

        # Compute M*M*z
        YtMz = numpy.matmul(Y.T, Mz)
        Binv_YtMz = numpy.matmul(Binv, YtMz)
        Y_Binv_YtMz = numpy.matmul(Y, Binv_YtMz)
        v = self.mixed_cor.solve(eta, Mz)
        MMz = v - Y_Binv_YtMz

        # Second derivative (only at the location of zero first derivative)
        zMz = numpy.dot(self.z, Mz)
        zM2z = numpy.dot(Mz, Mz)
        zM3z = numpy.dot(Mz, MMz)
        sigma2 = zMz / (n-m)
        # d2ell_deta2 = 0.5*(trace_M2 * zM2z - 2.0*trace_M * zM3z)

        # Warning: this relation is the second derivative only at optimal eta,
        # where the first derivative vanishes. It does not require the
        # computation of zM2z. But, for plotting, or using Hessian in
        # scipy.optimize.minimize, this formula must not be used, because it is
        # not the actual second derivative everywhere else other than optimal
        # point of eta.
        # d2ell_deta2 = (0.5/sigma2) * \
        #     ((trace_M2/(n-m) + (trace_M/(n-m))**2) * zMz - 2.0*zM3z)

        # This relation is the actual second derivative. Use this relation for
        # the Hessian in scipy.optimize.minimize.
        d2ell_deta2 = 0.5 * \
            (trace_M2 - 2.0*zM3z/sigma2 + zM2z**2/((n-m)*sigma2**2))

        # Return as scalar or array of length one
        if numpy.isscalar(hyperparam):
            return d2ell_deta2
        else:
            return numpy.array([[d2ell_deta2]], dtype=float)

    # =====================
    # likelihood der1 scale
    # =====================

    def _likelihood_der1_scale(self, hyperparam):
        """
        ell is the log likelihood probability. ell_dscale is d(ell)/d(theta),
        is the derivative of ell with respect to the distance scale (theta).
        """

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        # Set scale of the covariance object
        scale = self._hyperparam_to_scale(hyperparam[1:])
        self.mixed_cor.set_scale(scale)

        # Initialize jacobian
        dell_dscale = numpy.zeros((scale.size, ), dtype=float)

        # Find optimal sigma for the given eta.
        sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)

        n, m = self.X.shape

        # Computing Y=Kninv*X
        Y = self.mixed_cor.solve(eta, self.X)

        # B is Xt * Y
        B = numpy.matmul(self.X.T, Y)
        Binv = numpy.linalg.inv(B)

        # Compute Mz
        Mz = self.M_dot(Binv, Y, eta, self.z)

        # Needed to compute trace (TODO)
        Kn = self.mixed_cor.get_matrix(eta)
        Kninv = numpy.linalg.inv(Kn)

        # Knp is the derivative of mixed_cor (Kn) w.r.t p-th element of scale.
        for p in range(scale.size):

            # Compute zMSpMz
            KnpMz = self.mixed_cor.dot(eta, Mz, derivative=[p])
            zMKnpMz = numpy.dot(Mz, KnpMz)

            # Compute the first component of trace of Knp * M (TODO)
            Knp = self.mixed_cor.get_matrix(eta, derivative=[p])

            KnpKninv = Knp @ Kninv
            trace_KnpKninv, _ = imate.trace(KnpKninv, method='exact')
            # trace_KnpKninv = self.cov.traceinv(
            #         eta, Knp, imate_method='hutchinson')

            # Compute the second component of trace of Knp * M
            KnpY = self.mixed_cor.dot(eta, Y, derivative=[p])
            YtKnpY = numpy.matmul(Y.T, KnpY)
            BinvYtKnpY = numpy.matmul(Binv, YtKnpY)
            trace_BinvYtKnpY = numpy.trace(BinvYtKnpY)

            # Compute trace of Knp * M
            trace_KnpM = trace_KnpKninv - trace_BinvYtKnpY

            # Derivative of ell w.r.t p-th element of distance scale
            dell_dscale[p] = -0.5*trace_KnpM + 0.5*zMKnpMz / sigma**2

        return dell_dscale

    # =====================
    # likelihood der2 scale
    # =====================

    def _likelihood_der2_scale(self, hyperparam):
        """
        ell is the log likelihood probability. der2_scale is d2(ell)/d(theta2),
        is the second derivative of ell with respect to the distance scale
        (theta). The output is a 2D array of the size of scale.
        """

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        # Set scale of the covariance object
        scale = self._hyperparam_to_scale(hyperparam[1:])
        self.mixed_cor.set_scale(scale)

        # Initialize Hessian
        d2ell_dscale2 = numpy.zeros((scale.size, scale.size), dtype=float)

        # Find optimal sigma based on eta. Then compute sigma0
        sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)

        n, m = self.X.shape

        # Computing Y=Kninv*X
        Y = self.mixed_cor.solve(eta, self.X)

        # B is Xt * Y
        B = numpy.matmul(self.X.T, Y)
        Binv = numpy.linalg.inv(B)

        # Compute Mz
        Mz = self.M_dot(Binv, Y, eta, self.z)

        # Needed to compute trace (TODO)
        Kn = self.mixed_cor.get_matrix(eta)
        Kninv = numpy.linalg.inv(Kn)

        # Knp is the derivative of mixed_cor (Kn) w.r.t p-th element of scale.
        for p in range(scale.size):

            KnpMz = self.mixed_cor.dot(eta, Mz, derivative=[p])
            MKnpMz = self.M_dot(Binv, Y, eta, KnpMz)

            for q in range(scale.size):

                # 1. Compute zMSqMSpMz
                if p == q:
                    KnqMz = KnpMz
                else:
                    KnqMz = self.mixed_cor.dot(eta, Mz, derivative=[q])
                zMKnqMKnpMz = numpy.dot(KnqMz, MKnpMz)

                # 2. Compute zMKnpqMz
                KnpqMz = self.mixed_cor.dot(eta, Mz, derivative=[p, q])
                zMKnpqMz = numpy.dot(Mz, KnpqMz)

                # 3. Computing trace of Knpq * M in three steps

                # Compute the first component of trace of Knpq * Kninv (TODO)
                Knpq = self.mixed_cor.get_matrix(eta, derivative=[p, q])
                KnpqKninv = Knpq @ Kninv
                trace_KnpqKninv, _ = imate.trace(KnpqKninv, method='exact')

                # Compute the second component of trace of Knpq * M
                KnpqY = self.mixed_cor.dot(eta, Y, derivative=[p, q])
                YtKnpqY = numpy.matmul(Y.T, KnpqY)
                BinvYtKnpqY = numpy.matmul(Binv, YtKnpqY)
                trace_BinvYtKnpqY = numpy.trace(BinvYtKnpqY)

                # Compute trace of Knpq * M
                trace_KnpqM = trace_KnpqKninv - trace_BinvYtKnpqY

                # 4. Compute trace of Knp * M * Knq * M

                # Compute first part of trace of Knp * M * Knq * M
                Knp = self.mixed_cor.get_matrix(eta, derivative=[p])
                KnpKninv = Knp @ Kninv
                Knq = self.mixed_cor.get_matrix(eta, derivative=[q])
                if p == q:
                    KnqKninv = KnpKninv
                else:
                    KnqKninv = Knq @ Kninv
                KnpKninvKnqKninv = numpy.matmul(KnpKninv, KnqKninv)
                trace_KnpMKnqM_1, _ = imate.trace(KnpKninvKnqKninv,
                                                  method='exact')

                # Compute the second part of trace of Knp * M * Knq * M
                KnpY = Knp @ Y
                if p == q:
                    KnqY = KnpY
                else:
                    KnqY = Knq @ Y
                KninvKnqY = self.mixed_cor.solve(eta, KnqY)
                YtKnpKninvKnqY = numpy.matmul(KnpY.T, KninvKnqY)
                C21 = numpy.matmul(Binv, YtKnpKninvKnqY)
                C22 = numpy.matmul(Binv, YtKnpKninvKnqY.T)
                trace_KnpMKnqM_21 = numpy.trace(C21)
                trace_KnpMKnqM_22 = numpy.trace(C22)

                # Compute the third part of trace of Knp * M * Knq * M
                YtKnpY = numpy.matmul(Y.T, KnpY)
                if p == q:
                    YtKnqY = YtKnpY
                else:
                    YtKnqY = numpy.matmul(Y.T, KnqY)
                Dp = numpy.matmul(Binv, YtKnpY)
                if p == q:
                    Dq = Dp
                else:
                    Dq = numpy.matmul(Binv, YtKnqY)
                D = numpy.matmul(Dp, Dq)
                trace_KnpMKnqM_3 = numpy.trace(D)

                # Compute trace of Sp * M * Sq * M
                trace_KnpMKnqM = trace_KnpMKnqM_1 - trace_KnpMKnqM_21 - \
                    trace_KnpMKnqM_22 + trace_KnpMKnqM_3

                # 5. Second "local" derivatives w.r.t scale
                local_d2ell_dscale2 = -0.5*trace_KnpqM + 0.5*trace_KnpMKnqM + \
                    (0.5*zMKnpqMz - zMKnqMKnpMz) / sigma**2

                # Computing total second derivative
                dp_log_sigma2 = -numpy.dot(Mz, KnpMz) / ((n-m)*sigma**2)
                if p == q:
                    dq_log_sigma2 = dp_log_sigma2
                else:
                    dq_log_sigma2 = -numpy.dot(Mz, KnqMz) / ((n-m)*sigma**2)
                d2ell_dscale2[p, q] = local_d2ell_dscale2 + \
                    0.5 * (n-m) * dp_log_sigma2 * dq_log_sigma2

                if p != q:
                    d2ell_dscale2[q, p] = d2ell_dscale2[p, q]

        return d2ell_dscale2

    # =====================
    # likelihood der2 mixed
    # =====================

    def _likelihood_der2_mixed(self, hyperparam):
        """
        ell is the log likelihood probability. d2ell_deta_dscale is the mixed
        second derivative w.r.t eta and scale. The output is a 1D vector of the
        size of scale.
        """

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        # Set scale of the covariance object
        scale = self._hyperparam_to_scale(hyperparam[1:])
        self.mixed_cor.set_scale(scale)

        # Initialize mixed derivative as 2D array with one row.
        d2ell_deta_dscale = numpy.zeros((1, scale.size), dtype=float)

        # Find optimal sigma based on eta.
        sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)

        n, m = self.X.shape

        # Computing Y=Sinv*X.
        Y = self.mixed_cor.solve(eta, self.X)
        YtY = numpy.matmul(Y.T, Y)
        V = self.mixed_cor.solve(eta, Y)

        # B is Xt * Y
        B = numpy.matmul(self.X.T, Y)
        Binv = numpy.linalg.inv(B)

        # Compute Mz and MMz
        Mz = self.M_dot(Binv, Y, eta, self.z)
        MMz = self.M_dot(Binv, Y, eta, Mz)
        zMMz = numpy.dot(Mz, Mz)

        # Needed to compute trace (TODO)
        Kn = self.mixed_cor.get_matrix(eta)
        Kninv = numpy.linalg.inv(Kn)
        Kninv2 = Kninv @ Kninv

        # Knp is the derivative of mixed_cor (Kn) w.r.t p-th element of scale.
        for p in range(scale.size):

            # Compute zMKnpMMz
            KnpMz = self.mixed_cor.dot(eta, Mz, derivative=[p])
            zMKnpMz = numpy.dot(Mz, KnpMz)
            zMKnpMMz = numpy.dot(KnpMz, MMz)

            # Compute trace of KnpKninv2
            Knp = self.mixed_cor.get_matrix(eta, derivative=[p])
            KnpKninv2 = Knp @ Kninv2
            trace_KnpKninv2, _ = imate.trace(KnpKninv2, method='exact')

            # Compute traces
            KnpY = self.mixed_cor.dot(eta, Y, derivative=[p])
            YtKnpY = numpy.matmul(Y.T, KnpY)
            VtKnpY = numpy.matmul(V.T, KnpY)
            C1 = numpy.matmul(Binv, VtKnpY)
            C2 = numpy.matmul(Binv, VtKnpY.T)
            D1 = numpy.matmul(Binv, YtKnpY)
            D2 = numpy.matmul(Binv, YtY)
            D = numpy.matmul(D1, D2)

            trace_C1 = numpy.trace(C1)
            trace_C2 = numpy.trace(C2)
            trace_D = numpy.trace(D)

            # Compute trace of M * Sp * M
            trace_MKnpM = trace_KnpKninv2 - trace_C1 - trace_C2 + trace_D

            # Compute mixed derivative
            local_d2ell_deta_dscale = 0.5*trace_MKnpM - zMKnpMMz / sigma**2
            d2ell_deta_dscale[p] = local_d2ell_deta_dscale + \
                (0.5/((n-m)*sigma**4)) * zMMz * zMKnpMz

        return d2ell_deta_dscale

    # ===================
    # likelihood jacobian
    # ===================

    def likelihood_jacobian(self, sign_switch, hyperparam):
        """
        Computes Jacobian w.r.t eta, and if given, scale.
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

        # Compute first derivative w.r.t eta
        dell_deta = self._likelihood_der1_eta(hyperparam)

        # Because we use xi = log_eta instead of eta as the variable, the
        # derivative of ell w.r.t log_eta should be taken into account.
        if self.use_log_eta:
            eta = self._hyperparam_to_eta(hyperparam[0])
            dell_deta = dell_deta * eta * numpy.log(10.0)

        jacobian = dell_deta

        # Compute Jacobian w.r.t scale
        if hyperparam.size > 1:

            # Compute first derivative w.r.t scale
            dell_dscale = self._likelihood_der1_scale(hyperparam)

            # Convert derivative w.r.t log of scale
            if self.use_log_scale:
                scale = self._hyperparam_to_scale(hyperparam[1:])
                dell_dscale = numpy.multiply(dell_dscale, scale) * \
                    numpy.log(10.0)

            # Concatenate derivatives of eta and scale if needed
            jacobian = numpy.r_[dell_deta, dell_dscale]

        # Store Jacobian to member data (without sign-switch).
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
        Computes Hessian w.r.t eta, and if given, scale.
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

        # Compute second derivative w.r.t eta
        d2ell_deta2 = self._likelihood_der2_eta(hyperparam)

        # To convert derivative to log scale, Jacobian is needed. Note: The
        # Jacobian itself is already converted to log scale.
        if self.use_log_eta or self.use_log_scale:
            jacobian_ = self.likelihood_jacobian(False, hyperparam)

        # Because we use xi = log_eta instead of eta as the variable, the
        # derivative of ell w.r.t log_eta should be taken into account.
        if self.use_log_eta:
            eta = self._hyperparam_to_eta(hyperparam[0])
            dell_deta = jacobian_[0]

            # Convert second derivative to log scale (Note: dell_deta is
            # already in log scale)
            d2ell_deta2 = d2ell_deta2 * eta**2 * numpy.log(10.0)**2 + \
                dell_deta * numpy.log(10.0)

        # Hessian here is a 2D array of size 1.
        hessian = d2ell_deta2

        # Compute Hessian w.r.t scale
        if hyperparam.size > 1:

            # Compute second derivative w.r.t scale
            d2ell_dscale2 = self._likelihood_der2_scale(hyperparam)

            if self.use_log_scale:
                scale = self._hyperparam_to_scale(hyperparam[1:])
                dell_dscale = jacobian_[1:]

                for p in range(scale.size):
                    for q in range(scale.size):
                        if p == q:

                            # dell_dscale is already converted to logscale
                            d2ell_dscale2[p, q] = d2ell_dscale2[p, q] * \
                                scale[p]**2 * (numpy.log(10.0)**2) + \
                                dell_dscale * numpy.log(10.0)
                        else:
                            d2ell_dscale2[p, q] = d2ell_dscale2[p, q] * \
                                scale[p] * scale[q] * (numpy.log(10.0)**2)

            # Compute second mixed derivative w.r.t scale and eta
            d2ell_deta_dscale = self._likelihood_der2_mixed(hyperparam)

            if self.use_log_eta:
                eta = self._hyperparam_to_eta(hyperparam[0])
                d2ell_deta_dscale = d2ell_deta_dscale * eta * numpy.log(10.0)

            if self.use_log_scale:
                scale = self._hyperparam_to_scale(hyperparam[1:])
                for p in range(scale.size):
                    d2ell_deta_dscale[p] = d2ell_deta_dscale[p] * \
                        scale[p] * numpy.log(10.0)

            # Concatenate derivatives to form Hessian of all variables
            hessian = numpy.block(
                    [[d2ell_deta2, d2ell_deta_dscale],
                     [d2ell_deta_dscale.T, d2ell_dscale2]])

        # Store hessian to member data (without sign-switch).
        self.ell_hessian = hessian
        self.ell_hessian_hyperparam = hyperparam

        if sign_switch:
            hessian = -hessian

        return hessian

    # =========================
    # find optimal sigma sigma0
    # =========================

    def _find_optimal_sigma_sigma0(self, eta):
        """
        Based on a given eta, finds optimal sigma and sigma0.
        """

        if numpy.abs(eta) > self.max_eta:

            # eta is very large. Use Asymptotic relation
            sigma0 = self._find_optimal_sigma0()

            if numpy.isinf(eta):
                sigma = 0.
            else:
                sigma = sigma0 / numpy.sqrt(eta)

        else:

            # Find sigma
            sigma = self._find_optimal_sigma(eta)

            # Find sigma0
            if numpy.abs(eta) < self.min_eta:
                sigma0 = 0.0
            else:
                sigma0 = numpy.sqrt(eta) * sigma

        return sigma, sigma0

    # ==================
    # find optimal sigma
    # ==================

    def _find_optimal_sigma(self, eta):
        """
        When eta is *not* very large, finds optimal sigma.
        """

        Y = self.mixed_cor.solve(eta, self.X)
        w = self.mixed_cor.solve(eta, self.z)

        n, m = self.X.shape
        B = numpy.matmul(self.X.T, Y)
        Binv = numpy.linalg.inv(B)
        Ytz = numpy.matmul(Y.T, self.z)
        v = numpy.matmul(Y, numpy.matmul(Binv, Ytz))
        sigma2 = numpy.dot(self.z, w-v) / (n-m)
        sigma = numpy.sqrt(sigma2)

        return sigma

    # ===================
    # find optimal sigma0
    # ===================

    def _find_optimal_sigma0(self):
        """
        When eta is very large, we assume sigma is zero. Thus, sigma0 is
        computed by this function.
        """

        n, m = self.X.shape
        B = numpy.matmul(self.X.T, self.X)
        Binv = numpy.linalg.inv(B)
        Xtz = numpy.matmul(self.X.T, self.z)
        v = numpy.matmul(self.X, numpy.matmul(Binv, Xtz))
        sigma02 = numpy.dot(self.z, self.z-v) / (n-m)
        sigma0 = numpy.sqrt(sigma02)

        return sigma0

    # ==========================
    # find likelihood der1 zeros
    # ==========================

    def find_likelihood_der1_zeros(
            self,
            interval_eta,
            tol=1e-6,
            max_iterations=100,
            num_bracket_trials=3):
        """
        root finding of the derivative of ell.

        The log likelihood function is implicitly a function of eta. We have
        substituted the value of optimal sigma, which itself is a function of
        eta.
        """

        # Find an interval that the function changes sign before finding its
        # root (known as bracketing the function)
        log_eta_start = numpy.log10(interval_eta[0])
        log_eta_end = numpy.log10(interval_eta[1])

        # Initial points
        bracket = [log_eta_start, log_eta_end]
        bracket_found, bracket, bracket_values = \
            find_interval_with_sign_change(self._likelihood_der1_eta, bracket,
                                           num_bracket_trials, args=(), )

        if bracket_found:
            # There is a sign change in the interval of eta. Find root of ell
            # derivative

            # Find roots using Brent method
            # method = 'brentq'
            # res = scipy.optimize.root_scalar(self._likelihood_der1_eta,
            #                                  bracket=bracket, method=method,
            #                                  xtol=tol)
            # print('Iter: %d, Eval: %d, Converged: %s'
            #         % (res.iterations, res.function_calls, res.converged))

            # Find roots using Chandraputala method
            res = chandrupatla_method(self._likelihood_der1_eta, bracket,
                                      bracket_values, verbose=False, eps_m=tol,
                                      eps_a=tol, maxiter=max_iterations)

            # Extract results
            # eta = self._hyperparam_to_eta(res.root)   # Use with Brent
            eta = self._hyperparam_to_eta(res['root'])  # Use with Chandrupatla
            sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)
            iter = res['iterations']

            # Check second derivative
            # success = True
            # d2ell_deta2 = self._likelihood_der2_eta(eta)
            # if d2ell_deta2 < 0:
            #     success = True
            # else:
            #     success = False

        else:
            # bracket with sign change was not found.
            iter = 0

            # Evaluate the function in intervals
            eta_left = bracket[0]
            eta_right = bracket[1]
            dell_deta_left = bracket_values[0]
            dell_deta_right = bracket_values[1]

            # Second derivative of log likelihood at eta = zero, using either
            # of the two methods below:
            eta_zero = 0.0
            # method 1: directly from analytical equation
            d2ell_deta2_zero_eta = self._likelihood_der2_eta(eta_zero)

            # method 2: using forward differencing from first derivative
            # dell_deta_zero_eta = self._likelihood_der1_eta(
            #         numpy.log10(eta_zero))
            # d2ell_deta2_zero_eta = \
            #         (dell_deta_lowest_eta - dell_deta_zero_eta) / eta_lowest

            # print('dL/deta   at eta = 0.0:\t %0.2f'%dell_deta_zero_eta)
            print('dL/deta   at eta = %0.2e:\t %0.2f'
                  % (eta_left, dell_deta_left))
            print('dL/deta   at eta = %0.2e:\t %0.16f'
                  % (eta_right, dell_deta_right))
            print('d2L/deta2 at eta = 0.0:\t %0.2f'
                  % d2ell_deta2_zero_eta)

            # No sign change. Can not find a root
            if (dell_deta_left > 0) and (dell_deta_right > 0):
                if d2ell_deta2_zero_eta > 0:
                    eta = 0.0
                else:
                    eta = numpy.inf

            elif (dell_deta_left < 0) and (dell_deta_right < 0):
                if d2ell_deta2_zero_eta < 0:
                    eta = 0.0
                else:
                    eta = numpy.inf

            # Check eta
            if not (eta == 0 or numpy.isinf(eta)):
                raise ValueError('eta must be zero or inf at this point.')

            # Find sigma and sigma0
            sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)

        # Output dictionary
        result = {
            'hyperparam':
            {
                'sigma': sigma,
                'sigma0': sigma0,
                'eta': eta,
                'scale': None
            },
            'optimization':
            {
                'max_likelihood': None,
                'iter': iter
            }
        }

        return result

    # ===================
    # maximize likelihood
    # ===================

    def maximize_likelihood(
            self,
            tol=1e-3,
            hyperparam_guess=[0.1, 0.1],
            optimization_method='Nelder-Mead',
            verbose=False):
        """
        Maximizing the log-likelihood function over the space of parameters
        sigma and sigma0

        In this function, hyperparam = [sigma, sigma0].
        """

        # Keeping times
        initial_wall_time = time.time()
        initial_proc_time = time.process_time()

        # Convert eta to log of eta in hyperparam (if use_log_eta is True)
        eta_guess = hyperparam_guess[0]
        hyperparam_guess[0] = self._eta_to_hyperparam(eta_guess)

        # Convert scale to log of scale in hyperparam (if use_log_scale True)
        if (not numpy.isscalar(hyperparam_guess)) and \
                (len(hyperparam_guess) > 1):
            scale_guess = hyperparam_guess[1:]
            hyperparam_guess[1:] = self._scale_to_hyperparam(scale_guess)

        if optimization_method == 'chandrupatla':

            if (not numpy.isscalar(hyperparam_guess)) and \
                    (len(hyperparam_guess) > 1):

                warnings.warn('"chandrupatla" method does not optimize ' +
                              '"scale". The "distance scale in the given ' +
                              '"hyperparam_guess" will be ignored. To ' +
                              'optimize distance scale with "chandrupatla"' +
                              'method, set "profile_eta" to True.')

                if self.mixed_cor.get_scale() is None:
                    self.mixed_cor.set_scale(scale_guess)
                    warnings.warn('"scale" is set based on the guess value.')

            # Note: When using interpolation, make sure the interval below is
            # exactly the end points of eta_i, not less or more.
            min_eta_guess = numpy.min([1e-4, eta_guess * 1e-2])
            max_eta_guess = numpy.max([1e+3, eta_guess * 1e+2])
            interval_eta = [min_eta_guess, max_eta_guess]

            # Using root finding method on the first derivative w.r.t eta
            result = self._find_likelihood_der1_zeros(interval_eta)

            # Finding the maxima. This isn't necessary and affects run time
            result['optimization']['max_likelihood'] = self.likelihood(
                    False, result['hyperparam']['eta'])

            # The distance scale used in this method is the same as its guess.
            result['hyperparam']['scale'] = self.mixed_cor.get_scale()

        else:
            # Partial function of likelihood (with minus to make maximization
            # to a minimization).
            sign_switch = True
            likelihood_partial_func = partial(self.likelihood, sign_switch)

            # Partial function of Jacobian of likelihood (with minus sign)
            jacobian_partial_func = partial(self.likelihood_jacobian,
                                            sign_switch)

            # Partial function of Hessian of likelihood (with minus sign)
            hessian_partial_func = partial(self.likelihood_hessian,
                                           sign_switch)

            # Minimize
            res = scipy.optimize.minimize(likelihood_partial_func,
                                          hyperparam_guess,
                                          method=optimization_method, tol=tol,
                                          jac=jacobian_partial_func,
                                          hess=hessian_partial_func)

            # Extract res
            eta = self._hyperparam_to_eta(res.x[0])
            sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)
            max_ell = -res.fun

            # Distance scale
            if res.x.size > 1:
                scale = self._hyperparam_to_scale(res.x[1:])
            else:
                scale = self.mixed_cor.get_scale()

            # Output dictionary
            result = {
                'hyperparam':
                {
                    'sigma': sigma,
                    'sigma0': sigma0,
                    'eta': eta,
                    'scale': scale,
                },
                'optimization':
                {
                    'max_likelihood': max_ell,
                    'iter': res.nit,
                }
            }

        # Adding time to the results
        wall_time = time.time() - initial_wall_time
        proc_time = time.process_time() - initial_proc_time

        result['time'] = {
            'wall_time': wall_time,
            'proc_time': proc_time
        }

        return result

    # ============================
    # plot likelihood versus scale
    # ============================

    def plot_likelihood_versus_scale(
            self,
            result,
            other_etas=None):
        """
        Plots log likelihood versus sigma, eta hyperparam
        """

        # This function can only plot one dimensional data.
        dimension = self.mixed_cor.cor.dimension
        if dimension != 1:
            raise ValueError('To plot likelihood w.r.t "eta" and "scale", ' +
                             'the dimension of the data points should be one.')

        load_plot_settings()

        # Optimal point
        optimal_eta = result['hyperparam']['eta']

        # Convert eta to a numpy array
        if other_etas is not None:
            if numpy.isscalar(other_etas):
                other_etas = numpy.array([other_etas])
            elif isinstance(other_etas, list):
                other_etas = numpy.array(other_etas)
            elif not isinstance(other_etas, numpy.ndarray):
                raise TypeError('"other_etas" should be either a scalar, ' +
                                'list, or numpy.ndarray.')

        # Concatenate all given eta
        if other_etas is not None:
            etas = numpy.r_[optimal_eta, other_etas]
        else:
            etas = numpy.r_[optimal_eta]
        etas = numpy.sort(etas)

        # 2nd or 4th order finite difference coefficients for first derivative
        coeff = numpy.array([-1.0/2.0, 0.0, 1.0/2.0])
        # coeff = numpy.array([1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0])

        # The first axis of some arrays below is 3, used for varying eta for
        # numerical evaluating of mixed derivative w.r.t eta.
        stencil_size = coeff.size
        center_stencil = stencil_size//2  # Index of the center of stencil

        # Generate ell for various distance scales
        scale = numpy.logspace(-3, 2, 100)

        # The variable on the abscissa to take derivative with respect to it.
        if self.use_log_scale:
            scale_x = numpy.log10(scale)
        else:
            scale_x = scale

        d0_ell = numpy.zeros((stencil_size, etas.size, scale.size),
                             dtype=float)
        d1_ell = numpy.zeros((etas.size, scale.size), dtype=float)
        d2_ell = numpy.zeros((etas.size, scale.size), dtype=float)
        d2_mixed_ell = numpy.zeros((etas.size, scale.size),
                                   dtype=float)
        d1_ell_numerical = numpy.zeros((stencil_size, etas.size, scale.size-2),
                                       dtype=float)
        d2_ell_numerical = numpy.zeros((etas.size, scale.size-4),
                                       dtype=float)
        d2_mixed_ell_numerical = numpy.zeros((etas.size, scale.size-2),
                                             dtype=float)

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        colors = matplotlib.cm.nipy_spectral(
                numpy.linspace(0, 0.9, etas.size))

        for i in range(etas.size):

            # Stencil to perturb eta
            if self.use_log_eta:
                log_eta = numpy.log10(etas[i])
                d_eta = log_eta * 1e-3
                eta_stencil = 10.0**(
                        log_eta + d_eta *
                        numpy.arange(-stencil_size//2+1, stencil_size//2+1))
            else:
                d_eta = etas[i] * 1e-3
                eta_stencil = etas[i] + \
                    d_eta * numpy.arange(-stencil_size//2+1, stencil_size//2+1)

            for j in range(scale.size):

                # Set the scale
                self.mixed_cor.set_scale(scale[j])

                # Likelihood (first index, center_stencil, means the main etas)
                for k in range(stencil_size):
                    d0_ell[k, i, j] = self.likelihood(
                            False, self._eta_to_hyperparam(eta_stencil[k]))

                # First derivative of likelihood w.r.t distance scale
                sign_switch = False
                hyperparam = numpy.r_[
                        self._eta_to_hyperparam(etas[i]),
                        self._scale_to_hyperparam(scale[j])]
                d1_ell[i, j] = self.likelihood_jacobian(
                        sign_switch, hyperparam)[1]

                # Second derivative of likelihood w.r.t distance scale
                hessian_ = self.likelihood_hessian(sign_switch, hyperparam)
                d2_ell[i, j] = hessian_[1, 1]

                # Second mixed derivative of likelihood w.r.t distance scale
                d2_mixed_ell[i, j] = hessian_[0, 1]

            for k in range(stencil_size):
                # Compute first derivative numerically
                d1_ell_numerical[k, i, :] = \
                        (d0_ell[k, i, 2:] - d0_ell[k, i, :-2]) / \
                        (scale_x[2:] - scale_x[:-2])

                # Second mixed derivative numerically (finite difference)
                d2_mixed_ell_numerical[i, :] += \
                    coeff[k] * d1_ell_numerical[k, i, :] / d_eta

            # Compute second derivative numerically
            d2_ell_numerical[i, :] = \
                (d1_ell_numerical[center_stencil, i, 2:] -
                 d1_ell_numerical[center_stencil, i, :-2]) / \
                (scale_x[3:-1] - scale_x[1:-3])

            # Find maximum of ell
            max_index = numpy.argmax(d0_ell[center_stencil, i, :])
            optimal_scale = scale[max_index]
            optimal_ell = d0_ell[center_stencil, i, max_index]

            # Plot
            if etas[i] == optimal_eta:
                label = r'$\hat{\eta}=%0.2e$' % etas[i]
                marker = 'X'
            else:
                label = r'$\eta=%0.2e$' % etas[i]
                marker = 'o'
            ax[0, 0].plot(scale, d0_ell[center_stencil, i, :], color=colors[i],
                          label=label)
            ax[0, 1].plot(scale, d1_ell[i, :], color=colors[i], label=label)
            ax[1, 0].plot(scale, d2_ell[i, :], color=colors[i], label=label)
            ax[1, 1].plot(scale, d2_mixed_ell[i, :], color=colors[i],
                          label=label)
            ax[0, 1].plot(scale[1:-1], d1_ell_numerical[center_stencil, i, :],
                          '--', color=colors[i])
            ax[1, 0].plot(scale[2:-2], d2_ell_numerical[i, :], '--',
                          color=colors[i])
            ax[1, 1].plot(scale[1:-1], d2_mixed_ell_numerical[i, :], '--',
                          color=colors[i])
            p = ax[0, 0].plot(optimal_scale, optimal_ell, marker,
                              color=colors[i], markersize=3)
            ax[0, 1].plot(optimal_scale, 0.0,  marker, color=colors[i],
                          markersize=3)

        ax[0, 0].legend(p, [r'optimal $\theta$'])
        ax[0, 0].legend(loc='lower right')
        ax[0, 1].legend(loc='lower right')
        ax[1, 0].legend(loc='lower right')
        ax[1, 1].legend(loc='lower right')
        ax[0, 0].set_xscale('log')
        ax[0, 1].set_xscale('log')
        ax[1, 0].set_xscale('log')
        ax[1, 1].set_xscale('log')
        ax[0, 0].set_yscale('linear')
        ax[0, 1].set_yscale('linear')
        ax[1, 0].set_yscale('linear')
        ax[1, 1].set_yscale('linear')

        # Plot annotations
        ax[0, 0].set_xlim([scale[0], scale[-1]])
        ax[0, 1].set_xlim([scale[0], scale[-1]])
        ax[1, 0].set_xlim([scale[0], scale[-1]])
        ax[1, 1].set_xlim([scale[0], scale[-1]])
        ax[0, 0].set_xlabel(r'$\theta$')
        ax[0, 1].set_xlabel(r'$\theta$')
        ax[1, 0].set_xlabel(r'$\theta$')
        ax[1, 1].set_xlabel(r'$\theta$')
        ax[0, 0].set_ylabel(r'$\ell(\theta | \eta)$')

        if self.use_log_scale:
            ax[0, 1].set_ylabel(
                r'$\frac{\mathrm{d} \ell(\theta | \eta)}{\mathrm{d} ' +
                r'(\ln\theta)}$')
        else:
            ax[0, 1].set_ylabel(
                r'$\frac{\mathrm{d} \ell(\theta | \eta)}{\mathrm{d} \theta}$')

        if self.use_log_scale:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} ' +
                r'(\ln\theta)^2}$')
        else:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} ' +
                r'\theta^2}$')

        if self.use_log_scale and self.use_log_eta:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} ' +
                r'(\ln \theta) \mathrm{d} (\ln \eta)}$')
        elif self.use_log_scale:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} ' +
                r'(\ln\theta) \mathrm{d} \eta}$')
        elif self.use_log_eta:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} ' +
                r'\theta \mathrm{d} (\ln \eta)}$')
        else:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} ' +
                r'\theta \mathrm{d} \eta}$')

        ax[0, 0].set_title(r'Log likelihood function for fixed $\eta$')
        ax[0, 1].set_title(r'First derivative of log likelihood for fixed ' +
                           r'$\eta$')
        ax[1, 0].set_title(r'Second derivative of log likelihood for fixed ' +
                           r'$\eta$')
        ax[1, 1].set_title(r'Second mixed derivative of log likelihood for ' +
                           r'fixed $\eta$')
        ax[0, 0].grid(True, which='both')
        ax[0, 1].grid(True, which='both')
        ax[1, 0].grid(True, which='both')
        ax[1, 1].grid(True, which='both')

        plt.tight_layout()
        plt.show()

    # ==========================
    # plot likelihood versus eta
    # ==========================

    def plot_likelihood_versus_eta(
            self,
            result,
            other_scales=None):
        """
        Plots log likelihood versus sigma, eta hyperparam
        """

        # This function can only plot one dimensional data.
        dimension = self.mixed_cor.cor.dimension
        if dimension != 1:
            raise ValueError('To plot likelihood w.r.t "eta" and "scale", ' +
                             'the dimension of the data points should be one.')

        load_plot_settings()

        # Optimal point
        optimal_scale = result['hyperparam']['scale']

        # Convert scale to a numpy array
        if other_scales is not None:
            if numpy.isscalar(other_scales):
                other_scales = numpy.array([other_scales])
            elif isinstance(other_scales, list):
                other_scales = numpy.array(other_scales)
            elif not isinstance(other_scales, numpy.ndarray):
                raise TypeError('"other_scales" should be either a ' +
                                'scalar, list, or numpy.ndarray.')

        # Concatenate all given eta
        if other_scales is not None:
            scales = numpy.r_[optimal_scale, other_scales]
        else:
            scales = numpy.r_[optimal_scale]
        scales = numpy.sort(scales)

        # 2nd or 4th order finite difference coefficients for first derivative
        coeff = numpy.array([-1.0/2.0, 0.0, 1.0/2.0])
        # coeff = numpy.array([1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0])

        # The first axis of some arrays below is 3, used for varying the scale
        # for numerical evaluating of mixed derivative with respect to scale.
        stencil_size = coeff.size
        center_stencil = stencil_size//2  # Index of the center of stencil

        eta = numpy.logspace(-3, 3, 100)

        # The variable on the abscissa to take derivative with respect to it.
        if self.use_log_eta:
            x_eta = numpy.log10(eta)
        else:
            x_eta = eta

        d0_ell = numpy.zeros((stencil_size, scales.size, eta.size,),
                             dtype=float)
        d1_ell = numpy.zeros((scales.size, eta.size,), dtype=float)
        d2_ell = numpy.zeros((scales.size, eta.size,), dtype=float)
        d2_mixed_ell = numpy.zeros((scales.size, eta.size), dtype=float)
        d1_ell_numerical = numpy.zeros(
                (stencil_size, scales.size, eta.size-2,), dtype=float)
        d2_ell_numerical = numpy.zeros((scales.size, eta.size-4,), dtype=float)
        d2_mixed_ell_numerical = numpy.zeros((scales.size, eta.size-2),
                                             dtype=float)

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        colors = matplotlib.cm.nipy_spectral(
                numpy.linspace(0, 0.9, scales.size))

        for i in range(scales.size):

            # Stencil to perturb scale
            if self.use_log_scale:
                log_scale = numpy.log10(scales[i])
                d_scale = log_scale * 1e-3
                scale_stencil = 10.0**(
                        log_scale + d_scale *
                        numpy.arange(-stencil_size//2+1, stencil_size//2+1))
            else:
                d_scale = scales[i] * 1e-3
                scale_stencil = scales[i] + d_scale * \
                    numpy.arange(-stencil_size//2+1, stencil_size//2+1)

            # Iterate over the perturbations of scale
            for k in range(stencil_size):

                # Set the perturbed distance scale
                self.mixed_cor.set_scale(scale_stencil[k])

                for j in range(eta.size):

                    # Likelihood
                    d0_ell[k, i, j] = self.likelihood(
                            False, self._eta_to_hyperparam(eta[j]))

                    if k == center_stencil:

                        hyperparam = numpy.r_[
                                self._eta_to_hyperparam(eta[j]),
                                self._scale_to_hyperparam(scale_stencil[k])]

                        # First derivative w.r.t eta
                        sign_switch = False
                        d1_ell[i, j] = self.likelihood_jacobian(
                                sign_switch, hyperparam)[0]

                        # Second derivative w.r.t eta
                        hessian_ = self.likelihood_hessian(
                                sign_switch, hyperparam)
                        d2_ell[i, j] = hessian_[0, 0]

                        # Second mixed derivative w.r.t distance scale and eta
                        d2_mixed_ell[i, j] = hessian_[0, 1]

            for k in range(stencil_size):
                # Compute first derivative numerically
                d1_ell_numerical[k, i, :] = \
                    (d0_ell[k, i, 2:] - d0_ell[k, i, :-2]) / \
                    (x_eta[2:] - x_eta[:-2])

                # Second mixed derivative numerically (finite difference)
                d2_mixed_ell_numerical[i, :] += \
                    coeff[k] * d1_ell_numerical[k, i, :] / d_scale

            # Compute second derivative numerically
            d2_ell_numerical[i, :] = \
                (d1_ell_numerical[center_stencil, i, 2:] -
                 d1_ell_numerical[center_stencil, i, :-2]) / \
                (x_eta[3:-1] - x_eta[1:-3])

            # Find maximum of ell
            max_index = numpy.argmax(d0_ell[center_stencil, i, :])
            optimal_eta = eta[max_index]
            optimal_ell = d0_ell[center_stencil, i, max_index]

            if scales[i] == optimal_scale:
                label = r'$\hat{\theta} = %0.2e$' % scales[i]
                marker = 'X'
            else:
                label = r'$\theta = %0.2e$' % scales[i]
                marker = 'o'

            ax[0, 0].plot(eta, d0_ell[center_stencil, i, :], color=colors[i],
                          label=label)
            ax[0, 1].plot(eta, d1_ell[i, :], color=colors[i], label=label)
            ax[1, 0].plot(eta, d2_ell[i, :], color=colors[i], label=label)
            ax[1, 1].plot(eta, d2_mixed_ell[i, :], color=colors[i],
                          label=label)
            ax[0, 1].plot(eta[1:-1], d1_ell_numerical[center_stencil, i, :],
                          '--', color=colors[i])
            ax[1, 0].plot(eta[2:-2], d2_ell_numerical[i, :], '--',
                          color=colors[i])
            ax[1, 1].plot(eta[1:-1], d2_mixed_ell_numerical[i, :], '--',
                          color=colors[i])
            p = ax[0, 0].plot(optimal_eta, optimal_ell, marker,
                              color=colors[i], markersize=3)
            ax[0, 1].plot(optimal_eta, 0.0, marker, color=colors[i],
                          markersize=3)

        ax[0, 0].legend(p, [r'optimal $\eta$'])
        ax[0, 0].legend(loc='lower right')
        ax[0, 1].legend(loc='lower right')
        ax[1, 0].legend(loc='lower right')
        ax[1, 1].legend(loc='lower right')

        # Plot annotations
        ax[0, 0].set_xlim([eta[0], eta[-1]])
        ax[0, 1].set_xlim([eta[0], eta[-1]])
        ax[1, 0].set_xlim([eta[0], eta[-1]])
        ax[1, 1].set_xlim([eta[0], eta[-1]])
        ax[0, 0].set_xscale('log')
        ax[0, 1].set_xscale('log')
        ax[1, 0].set_xscale('log')
        ax[1, 1].set_xscale('log')
        ax[0, 0].set_yscale('linear')
        ax[0, 1].set_yscale('linear')
        ax[1, 0].set_yscale('linear')
        ax[1, 1].set_yscale('linear')
        ax[0, 0].set_xlabel(r'$\eta$')
        ax[0, 1].set_xlabel(r'$\eta$')
        ax[1, 0].set_xlabel(r'$\eta$')
        ax[1, 1].set_xlabel(r'$\eta$')
        ax[0, 0].set_ylabel(r'$\ell(\eta | \theta)$')

        if self.use_log_eta:
            ax[0, 1].set_ylabel(
                r'$\frac{\mathrm{d}\ell(\eta | \theta)}{\mathrm{d} ' +
                r'(\ln \eta)}$')
        else:
            ax[0, 1].set_ylabel(
                r'$\frac{\mathrm{d}\ell(\eta | \theta)}{\mathrm{d}\eta}$')

        if self.use_log_eta:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d} ' +
                r'(\ln \eta)^2}$')
        else:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d} \eta^2}$')

        if self.use_log_eta and self.use_log_scale:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d} ' +
                r'(\ln \eta) \mathrm{d} (\ln \theta)}$')
        elif self.use_log_eta:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d} ' +
                r'(\ln \eta) \mathrm{d} \theta}$')
        elif self.use_log_scale:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d}\eta ' +
                r'\mathrm{d} (\ln \theta)}$')
        else:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d}\eta ' +
                r'\mathrm{d} \theta}$')

        ax[0, 0].set_title(r'Log likelihood for fixed $\theta$')
        ax[0, 1].set_title(r'First derivative of log likelihood for ' +
                           r'fixed $\theta$')
        ax[1, 0].set_title(r'Second derivative of log likelihood for ' +
                           r'fixed $\theta$')
        ax[1, 1].set_title(r'Second mixed derivative of log likelihood for ' +
                           r'fixed $\theta$')
        ax[0, 0].grid(True, which='both')
        ax[0, 1].grid(True, which='both')
        ax[1, 0].grid(True, which='both')
        ax[1, 1].grid(True, which='both')

        plt.tight_layout()
        plt.show()

    # ================================
    # plot likelihood versus eta scale
    # ================================

    def plot_likelihood_versus_eta_scale(self, result):
        """
        Plots log likelihood versus sigma and eta hyperparam.
        """

        # This function can only plot one dimensional data.
        dimension = self.mixed_cor.cor.dimension
        if dimension != 1:
            raise ValueError('To plot likelihood w.r.t "eta" and "scale", ' +
                             'the dimension of the data points should be one.')

        load_plot_settings()

        # Optimal point
        optimal_eta = result['hyperparam']['eta']
        optimal_scale = result['hyperparam']['scale']
        optimal_ell = result['optimization']['max_likelihood']

        eta = numpy.logspace(-3, 3, 50)
        scale = numpy.logspace(-3, 2, 50)
        ell = numpy.zeros((scale.size, eta.size), dtype=float)

        # Compute ell
        for i in range(scale.size):
            self.mixed_cor.set_scale(scale[i])
            for j in range(eta.size):
                ell[i, j] = self.likelihood(
                        False, self._eta_to_hyperparam(eta[j]))

        # Convert inf to nan
        ell = numpy.where(numpy.isinf(ell), numpy.nan, ell)

        # Smooth data for finer plot
        # sigma_ = [2, 2]  # in unit of data pixel size
        # ell = scipy.ndimage.filters.gaussian_filter(
        #         ell, sigma_, mode='nearest')

        # Increase resolution for better contour plot
        N = 300
        f = scipy.interpolate.interp2d(
                numpy.log10(eta), numpy.log10(scale), ell, kind='cubic')
        scale_fine = numpy.logspace(numpy.log10(scale[0]),
                                    numpy.log10(scale[-1]), N)
        eta_fine = numpy.logspace(numpy.log10(eta[0]), numpy.log10(eta[-1]), N)
        x, y = numpy.meshgrid(eta_fine, scale_fine)
        ell_fine = f(numpy.log10(eta_fine), numpy.log10(scale_fine))

        # We will plot the difference of max of ell to ell, called z
        # max_ell = numpy.abs(numpy.max(ell_fine))
        # z = max_ell - ell_fine
        z = ell_fine

        # Cut data
        # cut_data = 0.92
        # clim = 0.87
        # z[z>CutData] = CutData   # Used for plotting data without prior

        # Min and max of data
        min_z = numpy.min(z)
        max_z = numpy.max(z)

        fig, ax = plt.subplots(ncols=3, figsize=(17, 5))

        # Adjust bounds of a colormap
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=2000):
            new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(
                    n=cmap.name, a=minval, b=maxval),
                cmap(numpy.linspace(minval, maxval, n)))
            return new_cmap

        # cmap = plt.get_cmap('gist_stern_r')
        # cmap = plt.get_cmap('rainbow_r')
        # cmap = plt.get_cmap('nipy_spectral_r')
        # cmap = plt.get_cmap('RdYlGn')
        # cmap = plt.get_cmap('ocean')
        # cmap = plt.get_cmap('gist_stern_r')
        # cmap = plt.get_cmap('RdYlBu')
        # cmap = plt.get_cmap('gnuplot_r')
        # cmap = plt.get_cmap('Spectral')
        cmap = plt.get_cmap('gist_earth')
        colormap = truncate_colormap(cmap, 0, 1)
        # colormap = truncate_colormap(cmap, 0.2, 0.9)  # for ocean

        # Contour fill Plot
        levels = numpy.linspace(min_z, max_z, 2000)
        c = ax[0].contourf(x, y, z, levels, cmap=colormap, zorder=-9)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(c, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel(r'$\ell(\eta, \theta)$')
        # c.set_clim(0, clim)
        # cbar.set_ticks([0,0.3,0.6,0.9,1])

        # Find max for each fixed eta
        opt_scale1 = numpy.zeros((eta_fine.size, ), dtype=float)
        opt_ell1 = numpy.zeros((eta_fine.size, ), dtype=float)
        opt_ell1[:] = numpy.nan
        for j in range(eta_fine.size):
            if numpy.all(numpy.isnan(ell_fine[:, j])):
                continue
            max_index = numpy.nanargmax(ell_fine[:, j])
            opt_scale1[j] = scale_fine[max_index]
            opt_ell1[j] = ell_fine[max_index, j]
        ax[0].plot(eta_fine, opt_scale1, color='red',
                   label=r'$\hat{\theta}(\eta)$')
        ax[1].plot(eta_fine, opt_ell1, color='red')

        # Find max for each fixed scale
        opt_eta2 = numpy.zeros((scale_fine.size, ), dtype=float)
        opt_ell2 = numpy.zeros((scale_fine.size, ), dtype=float)
        opt_ell2[:] = numpy.nan
        for i in range(scale_fine.size):
            if numpy.all(numpy.isnan(ell_fine[i, :])):
                continue
            max_index = numpy.nanargmax(ell_fine[i, :])
            opt_eta2[i] = eta_fine[max_index]
            opt_ell2[i] = ell_fine[i, max_index]
        ax[0].plot(opt_eta2, scale_fine, color='black',
                   label=r'$\hat{\eta}(\theta)$')
        ax[2].plot(scale_fine, opt_ell2, color='black')

        # Plot max of the whole 2D array
        max_indices = numpy.unravel_index(numpy.nanargmax(ell_fine),
                                          ell_fine.shape)
        opt_scale = scale_fine[max_indices[0]]
        opt_eta = eta_fine[max_indices[1]]
        opt_ell = ell_fine[max_indices[0], max_indices[1]]
        ax[0].plot(opt_eta, opt_scale, 'o', color='red', markersize=6,
                   label=r'$(\hat{\eta}, \hat{\theta})$ (by brute force on ' +
                         r'grid)')
        ax[1].plot(opt_eta, opt_ell, 'o', color='red',
                   label=r'$\ell(\hat{\eta}, \hat{\theta})$ (by brute force ' +
                         r'on grid)')
        ax[2].plot(opt_scale, opt_ell, 'o', color='red',
                   label=r'$\ell(\hat{\eta}, \hat{\theta})$ (by brute force ' +
                         r'on grid)')

        # Plot optimal point as found by the profile likelihood method
        ax[0].plot(optimal_eta, optimal_scale, 'o', color='black',
                   markersize=6,
                   label=r'$\max_{\eta, \theta} \ell$ (by optimization)')
        ax[1].plot(optimal_eta, optimal_ell, 'o', color='black',
                   label=r'$\ell(\hat{\eta}, \hat{\theta})$ (by optimization)')
        ax[2].plot(optimal_scale, optimal_ell, 'o', color='black',
                   label=r'$\ell(\hat{\eta}, \hat{\theta})$ (by optimization)')

        # Plot annotations
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[0].set_xlim([eta[0], eta[-1]])
        ax[1].set_xlim([eta[0], eta[-1]])
        ax[0].set_ylim([scale[0], scale[-1]])
        ax[2].set_xlim([scale[0], scale[-1]])
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[2].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xlabel(r'$\log_{10}(\eta)$')
        ax[1].set_xlabel(r'$\log_{10}(\eta)$')
        ax[2].set_xlabel(r'$\log_{10}(\theta)$')
        ax[0].set_ylabel(r'$\log_{10}(\theta)$')
        ax[1].set_ylabel(r'$\ell(\eta, \hat{\theta}(\eta))$')
        ax[2].set_ylabel(r'$\ell(\hat{\eta}(\theta), \theta)$')
        ax[0].set_title('Log likelihood function')
        ax[1].set_title(r'Log Likelihood profiled over $\theta$ ')
        ax[2].set_title(r'Log likelihood profiled over $\eta$')
        ax[1].grid(True)
        ax[2].grid(True)

        plt.tight_layout()
        plt.show()

    # =======================
    # compute bounds der1 eta
    # =======================

    def _compute_bounds_der1_eta(self, K, eta):
        """
        Upper and lower bound.
        """

        n, m = self.X.shape
        eigenvalue_smallest = scipy.linalg.eigh(K, eigvals_only=True,
                                                check_finite=False,
                                                subset_by_index=[0, 0])[0]

        eigenvalue_largest = scipy.linalg.eigh(K, eigvals_only=True,
                                               check_finite=False,
                                               subset_by_index=[n-1, n-1])[0]
        # print('Eigenvalues of K:')
        # print(eigenvalue_smallest)
        # print(eigenvalue_largest)
        dell_deta_upper_bound = 0.5*(n-m) * \
            (1/(eta+eigenvalue_smallest) - 1/(eta+eigenvalue_largest))
        dell_deta_lower_bound = -dell_deta_upper_bound

        return dell_deta_upper_bound, dell_deta_lower_bound

    # ==========================
    # compute asymptote der1 eta
    # ==========================

    def _compute_asymptote_der1_eta(self, K, eta):
        """
        Computes first and second order asymptote to the first derivative of
        log marginal likelihood function.
        """

        # Initialize output
        asymptote_1_order = numpy.empty(eta.size)
        asymptote_2_order = numpy.empty(eta.size)

        n, m = self.X.shape
        I = numpy.eye(n)                                           # noqa: E741
        # Im = numpy.eye(m)
        Q = self.X @ numpy.linalg.inv(self.X.T @ self.X) @ self.X.T
        R = I - Q
        N = K @ R
        N2 = N @ N
        N3 = N2 @ N
        N4 = N3 @ N

        mtrN = numpy.trace(N)/(n-m)
        mtrN2 = numpy.trace(N2)/(n-m)

        A0 = -R @ (mtrN*I - N)
        A1 = R @ (mtrN*N + mtrN2*I - 2*N2)
        A2 = -R @ (mtrN*N2 + mtrN2*N - 2*N3)
        A3 = R @ (mtrN2*N2 - N4)

        zRz = numpy.dot(self.z, numpy.dot(R, self.z))
        z_Rnorm = numpy.sqrt(zRz)
        zc = self.z / z_Rnorm

        a0 = numpy.dot(zc, numpy.dot(A0, zc))
        a1 = numpy.dot(zc, numpy.dot(A1, zc))
        a2 = numpy.dot(zc, numpy.dot(A2, zc))
        a3 = numpy.dot(zc, numpy.dot(A3, zc))

        for i in range(eta.size):

            asymptote_1_order[i] = (-0.5*(n-m)) * (a0 + a1/eta[i])/eta[i]**2
            asymptote_2_order[i] = (-0.5*(n-m)) * \
                (a0 + a1/eta[i] + a2/eta[i]**2 + a3/eta[i]**3)/eta[i]**2

        # Roots
        polynomial_1 = numpy.array([a0, a1])
        polynomial_2 = numpy.array([a0, a1, a2, a3])

        roots_1 = numpy.roots(polynomial_1)
        roots_2 = numpy.roots(polynomial_2)

        # Remove complex roots
        roots_2 = numpy.sort(numpy.real(
            roots_2[numpy.abs(numpy.imag(roots_2)) < 1e-10]))

        print('asymptote roots:')
        print(roots_1)
        print(roots_2)

        return asymptote_1_order, asymptote_2_order, roots_1, roots_2

    # ========================
    # plot likelihood der1 eta
    # ========================

    def plot_likelihood_der1_eta(self, result):
        """
        Plots the derivative of log likelihood as a function of eta.
        Also it shows where the optimal eta is, which is the location
        where the derivative is zero.
        """

        load_plot_settings()

        # Optimal point
        optimal_eta = result['hyperparam']['eta']

        if (optimal_eta != 0) and (not numpy.isinf(optimal_eta)):
            plot_optimal_eta = True
        else:
            plot_optimal_eta = False

        # Specify which portion of eta array be high resolution for plotting
        # in the inset axes
        log_eta_start = -3
        log_eta_end = 3

        if plot_optimal_eta:
            log_eta_start_high_res = numpy.floor(numpy.log10(optimal_eta))
            log_eta_end_high_res = log_eta_start_high_res + 2

            # Arrays of low and high resolutions of eta
            eta_high_res = numpy.logspace(log_eta_start_high_res,
                                          log_eta_end_high_res, 100)
            eta_low_res_left = numpy.logspace(log_eta_start,
                                              log_eta_start_high_res, 50)
            eta_low_res_right = numpy.logspace(log_eta_end_high_res,
                                               log_eta_end, 20)

            # array of eta as a mix of low and high res
            if log_eta_end_high_res >= log_eta_end:
                eta = numpy.r_[eta_low_res_left, eta_high_res]
            else:
                eta = numpy.r_[eta_low_res_left, eta_high_res,
                               eta_low_res_right]

        else:
            eta = numpy.logspace(log_eta_start, log_eta_end, 100)

        # Compute derivative of L
        dell_deta = numpy.zeros(eta.size)
        for i in range(eta.size):
            dell_deta[i] = self._likelihood_der1_eta(
                    self._eta_to_hyperparam(eta[i]))

        # Compute upper and lower bound of derivative
        K = self.mixed_cor.get_matrix(0.0)
        dell_deta_upper_bound, dell_deta_lower_bound = \
            self._compute_bounds_der1_eta(K, eta)

        # Compute asymptote of first derivative, using both first and second
        # order approximation
        try:
            # eta_high_res might not be defined, depending on plot_optimal_eta
            x = eta_high_res
        except NameError:
            x = numpy.logspace(1, log_eta_end, 100)
        dell_deta_asymptote_1, dell_deta_asymptote_2, roots_1, roots_2 = \
            self._compute_asymptote_der1_eta(K, x)

        # Main plot
        fig, ax1 = plt.subplots()
        ax1.semilogx(eta, dell_deta_upper_bound, '--', color='black',
                     label='Upper bound')
        ax1.semilogx(eta, dell_deta_lower_bound, '-.', color='black',
                     label='Lower bound')
        ax1.semilogx(eta, dell_deta, color='black', label='Exact')
        if plot_optimal_eta:
            ax1.semilogx(optimal_eta, 0, '.', marker='o', markersize=4,
                         color='black')

        # Min of plot limit
        # ax1.set_yticks(numpy.r_[numpy.arange(-120, 1, 40), 20])
        max_plot = numpy.max(dell_deta)
        max_plot_lim = numpy.ceil(numpy.abs(max_plot)/10.0) * \
            10.0*numpy.sign(max_plot)

        min_plot_lim1 = -100
        ax1.set_yticks(numpy.array([min_plot_lim1, 0, max_plot_lim]))
        ax1.set_ylim([min_plot_lim1, max_plot_lim])
        ax1.set_xlim([eta[0], eta[-1]])
        ax1.set_xlabel(r'$\eta$')
        ax1.set_ylabel(r'$\mathrm{d} \ell_{\hat{\sigma}^2(\eta)}' +
                       r'(\eta)/\mathrm{d} \eta$')
        ax1.set_title('Derivative of Log Marginal Likelihood Function')
        ax1.grid(True)
        # ax1.legend(loc='upper left', frameon=False)
        ax1.patch.set_facecolor('none')

        # Inset plot
        if plot_optimal_eta:
            ax2 = plt.axes([0, 0, 1, 1])
            # Manually set position and relative size of inset axes within ax1
            ip = InsetPosition(ax1, [0.43, 0.39, 0.5, 0.5])
            ax2.set_axes_locator(ip)
            # Mark the region corresponding to the inset axes on ax1 and draw
            # lines in grey linking the two axes.

            # Avoid inset mark lines intersect inset axes by setting its anchor
            if log_eta_end > log_eta_end_high_res:
                mark_inset(ax1, ax2, loc1=3, loc2=4, facecolor='none',
                           edgecolor='0.5')
            else:
                mark_inset(ax1, ax2, loc1=3, loc2=1, facecolor='none',
                           edgecolor='0.5')

            ax2.semilogx(eta, numpy.abs(dell_deta_upper_bound), '--',
                         color='black')
            ax2.semilogx(eta, numpy.abs(dell_deta_lower_bound), '-.',
                         color='black')
            ax2.semilogx(x, dell_deta_asymptote_1,
                         label=r'$1^{\text{st}}$ order asymptote',
                         color='chocolate')
            ax2.semilogx(x, dell_deta_asymptote_2,
                         label=r'$2^{\text{nd}}$ order asymptote',
                         color='olivedrab')
            ax2.semilogx(eta_high_res,
                         dell_deta[eta_low_res_left.size:
                                   eta_low_res_left.size+eta_high_res.size],
                         color='black')
            ax2.semilogx(optimal_eta, 0, marker='o', markersize=6, linewidth=0,
                         color='white', markerfacecolor='black',
                         label=r'Exact root at $\hat{\eta}_{\phantom{2}} ' +
                               r'= 10^{%0.2f}$' % numpy.log10(optimal_eta))
            ax2.semilogx(roots_1[-1], 0, marker='o', markersize=6, linewidth=0,
                         color='white', markerfacecolor='chocolate',
                         label=r'Approximated root at $\hat{\eta}_1 = ' +
                               r'10^{%0.2f}$' % numpy.log10(roots_1[-1]))
            ax2.semilogx(roots_2[-1], 0, marker='o', markersize=6, linewidth=0,
                         color='white', markerfacecolor='olivedrab',
                         label=r'Approximated root at $\hat{\eta}_2 = ' +
                               r'10^{%0.2f}$' % numpy.log10(roots_2[-1]))
            ax2.set_xlim([eta_high_res[0], eta_high_res[-1]])
            # plt.setp(ax2.get_yticklabels(), backgroundcolor='white')

            # Find suitable range for plot limits
            min_plot = numpy.abs(numpy.min(dell_deta))
            min_plot_base = 10**numpy.floor(numpy.log10(numpy.abs(min_plot)))
            # min_plot_lim = numpy.ceil(min_plot/min_plot_base)*min_plot_base
            min_plot_lim = numpy.ceil(min_plot/min_plot_base + 1.0) * \
                min_plot_base
            ax2.set_ylim([-min_plot_lim, min_plot_lim])
            ax2.set_yticks([-numpy.abs(min_plot_lim), 0,
                            numpy.abs(min_plot_lim)])

            ax2.text(optimal_eta*10**0.05, min_plot_lim*0.05,
                     r'$\hat{\eta}$' % numpy.log10(optimal_eta),
                     horizontalalignment='left', verticalalignment='bottom',
                     fontsize=10)
            ax2.text(roots_1[-1]*10**0.05, min_plot_lim*0.05,
                     r'$\hat{\eta}_1$' % numpy.log10(optimal_eta),
                     horizontalalignment='left', verticalalignment='bottom',
                     fontsize=10)
            ax2.text(roots_2[-1]*10**0.05, min_plot_lim*0.05,
                     r'$\hat{\eta}_2$' % numpy.log10(optimal_eta),
                     horizontalalignment='left', verticalalignment='bottom',
                     fontsize=10)
            # ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax2.grid(True, axis='y')
            ax2.set_facecolor('oldlace')
            plt.setp(ax2.get_xticklabels(), backgroundcolor='white')
            ax2.tick_params(axis='x', labelsize=10)
            ax2.tick_params(axis='y', labelsize=10)

            # ax2.set_yticklabels(ax2.get_yticks(), backgroundcolor='w')
            # ax2.tick_params(axis='y', which='major', pad=0)

        handles, labels = [], []
        for ax in [ax1, ax2]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        plt.legend(handles, labels, frameon=False, fontsize='small',
                   loc='upper left', bbox_to_anchor=(1.2, 1.04))

        # Save plots
        # plt.tight_layout()
        filename = 'likelihood_first_derivative'
        save_plot(plt, filename, transparent_background=False, pdf=True)

        plt.show()
