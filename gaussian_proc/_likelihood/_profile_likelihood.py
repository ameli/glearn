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
from ._likelihood_utilities import M_dot
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
        self.cov = cov
        self.mixed_cor = self.cov.mixed_cor

        # Member data
        self.use_logscale = True
        self.hyperparam = None
        self.lp = None
        self.dlp_deta = None
        self.d2lp_deta2 = None

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

        sign_switch changes the sign of the output from lp to -lp. When True,
        this is used to minimizing (instead of maximizing) the negative of
        log-likelihood function.
        """

        if (not numpy.isscalar(hyperparam)) and (hyperparam.size > 1):
            scale = numpy.abs(hyperparam[1:])
            self.cov.set_scale(scale)

        # Get log_eta
        if numpy.isscalar(hyperparam):
            log_eta = hyperparam
        else:
            log_eta = hyperparam[0]

        # Change log_eta to eta
        if numpy.isneginf(log_eta):
            eta = 0.0
        else:
            eta = 10.0**log_eta
        # eta = numpy.abs(log_eta)  # Test

        n, m = self.X.shape

        max_eta = 1e+16
        if numpy.abs(eta) >= max_eta:

            B = numpy.matmul(self.X.T, self.X)
            Binv = numpy.linalg.inv(B)
            logdet_Binv = numpy.log(numpy.linalg.det(Binv))

            # Optimal sigma0 when eta is very large
            sigma0 = self._find_optimal_sigma0()

            # Log likelihood
            lp = -0.5*(n-m)*numpy.log(2.0*numpy.pi) \
                - (n-m)*numpy.log(sigma0) - 0.5*logdet_Binv - 0.5*(n-m)

        else:

            sigma = self._find_optimal_sigma(eta)
            logdet_Kn = self.mixed_cor.logdet(eta)

            # Compute log det (X.T Kn_inv X)
            Y = self.mixed_cor.solve(eta, self.X)

            XtKninvX = numpy.matmul(self.X.T, Y)
            logdet_XtKninvX = numpy.log(numpy.linalg.det(XtKninvX))

            # Log likelihood
            lp = -0.5*(n-m)*numpy.log(2.0*numpy.pi) \
                - (n-m)*numpy.log(sigma) - 0.5*logdet_Kn \
                - 0.5*logdet_XtKninvX \
                - 0.5*(n-m)

        # If lp is used in scipy.optimize.minimize, change the sign to obtain
        # the minimum of -lp
        if sign_switch:
            lp = -lp

        return lp

    # ===================
    # likelihood der1 eta
    # ===================

    def _likelihood_der1_eta(self, hyperparam):
        """
        lp is the log likelihood probability. lp_deta is d(lp)/d(eta), is the
        derivative of lp with respect to eta when the optimal value of sigma is
        substituted in the likelihood function per given eta.
        """

        # hyperparam
        if numpy.isscalar(hyperparam):
            log_eta = hyperparam
        else:
            log_eta = hyperparam[0]

        # Change log_eta to eta
        if numpy.isneginf(log_eta):
            eta = 0.0
        else:
            eta = 10.0**log_eta
        # eta = numpy.abs(log_eta)  # Test

        # Include derivative w.r.t scale
        if (not numpy.isscalar(hyperparam)) and (hyperparam.size > 1):
            scale = numpy.abs(hyperparam[1:])
            self.cov.set_scale(scale)

        # Compute Kn_inv*X and Kn_inv*z
        Y = self.mixed_cor.solve(eta, self.X)
        w = self.mixed_cor.solve(eta, self.z)

        n, m = self.X.shape

        # Splitting M into M1 and M2. Here, we compute M2
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
        dlp_deta = -0.5*(trace_M - zM2z/sigma2)

        # Because we use xi = log_eta instead of eta as the variable, the
        # derivative of lp w.r.t log_eta is dlp_deta * deta_dxi, and
        # deta_dxi is eta * lob(10).
        # dlp_deta = dlp_deta * eta * numpy.log(10.0)

        # Test
        # log_eta = hyperparam[0]
        # eta = 10.0**log_eta
        # dlp_deta *= eta * numpy.log(10.0)

        # Return as scalar or array of length one
        if numpy.isscalar(hyperparam):
            return dlp_deta
        else:
            return numpy.array([dlp_deta], dtype=float)

    # ===================
    # likelihood der2 eta
    # ===================

    def _likelihood_der2_eta(self, hyperparam):
        """
        The second derivative of lp is computed as a function of only eta.
        Here, we substituted optimal value of sigma, which is self is a
        function of eta.
        """

        # hyperparam
        if numpy.isscalar(hyperparam):
            log_eta = hyperparam
        else:
            log_eta = hyperparam[0]

        # Change log_eta to eta
        if numpy.isneginf(log_eta):
            eta = 0.0
        else:
            eta = 10.0**log_eta

        # Include derivative w.r.t scale
        if (not numpy.isscalar(hyperparam)) and (hyperparam.size > 1):
            scale = numpy.abs(hyperparam[1:])
            self.cov.set_scale(scale)

        Y = self.mixed_cor.solve(eta, self.X)
        V = self.mixed_cor.solve(eta, Y)
        w = self.mixed_cor.solve(eta, self.z)

        n, m = self.X.shape

        # Splitting M
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

        # Find z.T * M**3 * z
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
        # d2lp_deta2 = 0.5*(trace_M2 * zM2z - 2.0*trace_M * zM3z)

        # Warning: this relation is the second derivative only at optimal eta,
        # where the first derivative vanishes. It does not require the
        # computation of zM2z. But, for plotting, or using hessian in
        # scipy.optimize.minimize, this formula must not be used, because it is
        # not the actual second derivative everywhere else other than optimal
        # point of eta.
        # d2lp_deta2 = (0.5/sigma2) * \
        #     ((trace_M2/(n-m) + (trace_M/(n-m))**2) * zMz - 2.0*zM3z)

        # This relation is the actual second derivative. Use this relation for
        # the hessian in scipy.optimize.minimize.
        d2lp_deta2 = 0.5 * \
            (trace_M2 - 2.0*zM3z/sigma2 + zM2z**2/((n-m)*sigma2**2))

        # Return as scalar or array of length one
        if numpy.isscalar(hyperparam):
            return d2lp_deta2
        else:
            return numpy.array([[d2lp_deta2]], dtype=float)

    # =====================
    # likelihood der1 scale
    # =====================

    def _likelihood_der1_scale(self, hyperparam):
        """
        lp is the log likelihood probability. lp_dscale is d(lp)/d(theta), is
        the derivative of lp with respect to the distance scale (theta).
        """

        # Get log_eta
        if numpy.isscalar(hyperparam):
            log_eta = hyperparam
        else:
            log_eta = hyperparam[0]
        eta = 10.0**log_eta
        scale = numpy.abs(hyperparam[1:])
        self.cov.set_scale(scale)

        # Initialize jacobian
        der1_scale = numpy.zeros((scale.size, ), dtype=float)

        # Find optimal sigma based on eta. Then compute sigma0
        sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)

        n, m = self.X.shape

        # Computing Y=Sinv*X and w=Sinv*z.
        Y = self.cov.solve(sigma, sigma0, self.X)

        # B is Xt * Y
        B = numpy.matmul(self.X.T, Y)
        Binv = numpy.linalg.inv(B)

        # Compute Mz
        Mz = M_dot(self.cov, Binv, Y, sigma, sigma0, self.z)

        # Needed to compute trace (TODO)
        S = self.cov.get_matrix(sigma, sigma0)
        Sinv = numpy.linalg.inv(S)

        # Sp is the derivative of cov w.r.t the p-th element of scale.
        for p in range(scale.size):

            # Compute zMSpMz
            SpMz = self.cov.dot(sigma, sigma0, Mz, derivative=[p])
            zMSpMz = numpy.dot(Mz, SpMz)

            # Compute the first component of trace of Sp * M (TODO)
            Sp = self.cov.get_matrix(sigma, sigma0, derivative=[p])

            SpSinv = Sp @ Sinv
            trace_SpSinv, _ = imate.trace(SpSinv, method='exact')
            # trace_SpSinv = self.cov.traceinv(sigma, sigma0, Sp,
            #                                  imate_method='hutchinson')

            # Compute the second component of trace of Sp * M
            SpY = self.cov.dot(sigma, sigma0, Y, derivative=[p])
            YtSpY = numpy.matmul(Y.T, SpY)
            BinvYtSpY = numpy.matmul(Binv, YtSpY)
            trace_BinvYtSpY = numpy.trace(BinvYtSpY)

            # Compute trace of Sp * M
            trace_SpM = trace_SpSinv - trace_BinvYtSpY

            # Derivative of lp w.r.t p-th element of distance scale
            der1_scale[p] = -0.5*trace_SpM + 0.5*zMSpMz

        return der1_scale

    # =====================
    # likelihood der2 scale
    # =====================

    def _likelihood_der2_scale(self, hyperparam):
        """
        lp is the log likelihood probability. der2_scale is d2(lp)/d(theta2),
        is the second derivative of lp with respect to the distance scale
        (theta). The output is a 2D array of the size of scale.
        """

        # Get log_eta
        if numpy.isscalar(hyperparam):
            log_eta = hyperparam
        else:
            log_eta = hyperparam[0]
        eta = 10.0**log_eta
        scale = numpy.abs(hyperparam[1:])
        self.cov.set_scale(scale)

        # Initialize hessian
        der2_scale = numpy.zeros((scale.size, scale.size), dtype=float)

        # Find optimal sigma based on eta. Then compute sigma0
        sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)

        n, m = self.X.shape

        # Computing Y=Sinv*X and w=Sinv*z.
        Y = self.cov.solve(sigma, sigma0, self.X)

        # B is Xt * Y
        B = numpy.matmul(self.X.T, Y)
        Binv = numpy.linalg.inv(B)

        # Compute Mz
        Mz = M_dot(self.cov, Binv, Y, sigma, sigma0, self.z)

        # Needed to compute trace (TODO)
        S = self.cov.get_matrix(sigma, sigma0)
        Sinv = numpy.linalg.inv(S)

        # Sp is the derivative of cov w.r.t the p-th element of scale.
        for p in range(scale.size):

            SpMz = self.cov.dot(sigma, sigma0, Mz, derivative=[p])
            MSpMz = M_dot(self.cov, Binv, Y, sigma, sigma0, SpMz)

            for q in range(scale.size):

                # 1. Compute zMSqMSpMz
                if p == q:
                    SqMz = SpMz
                else:
                    SqMz = self.cov.dot(sigma, sigma0, Mz, derivative=[q])
                zMSqMSpMz = numpy.dot(SqMz, MSpMz)

                # 2. Compute zMSpqMz
                SpqMz = self.cov.dot(sigma, sigma0, Mz, derivative=[p, q])
                zMSpqMz = numpy.dot(Mz, SpqMz)

                # 3. Computing trace of Spq * M in three steps

                # Compute the first component of trace of Spq * Sinv (TODO)
                Spq = self.cov.get_matrix(sigma, sigma0, derivative=[p, q])
                SpqSinv = Spq @ Sinv
                trace_SpqSinv, _ = imate.trace(SpqSinv, method='exact')

                # Compute the second component of trace of Spq * M
                SpqY = self.cov.dot(sigma, sigma0, Y, derivative=[p, q])
                YtSpqY = numpy.matmul(Y.T, SpqY)
                BinvYtSpqY = numpy.matmul(Binv, YtSpqY)
                trace_BinvYtSpqY = numpy.trace(BinvYtSpqY)

                # Compute trace of Spq * M
                trace_SpqM = trace_SpqSinv - trace_BinvYtSpqY

                # 4. Compute trace of Sp * M * Sq * M

                # Compute first part of trace of Sp * M * Sq * M
                Sp = self.cov.get_matrix(sigma, sigma0, derivative=[p])
                SpSinv = Sp @ Sinv
                Sq = self.cov.get_matrix(sigma, sigma0, derivative=[q])
                if p == q:
                    SqSinv = SpSinv
                else:
                    SqSinv = Sq @ Sinv
                SpSinvSqSinv = numpy.matmul(SpSinv, SqSinv)
                trace_SpMSqM_1, _ = imate.trace(SpSinvSqSinv,
                                                method='exact')

                # Compute the second part of trace of Sp * M * Sq * M
                SpY = Sp @ Y
                if p == q:
                    SqY = SpY
                else:
                    SqY = Sq @ Y
                SinvSqY = self.cov.solve(sigma, sigma0, SqY)
                YtSpSinvSqY = numpy.matmul(SpY.T, SinvSqY)
                C21 = numpy.matmul(Binv, YtSpSinvSqY)
                C22 = numpy.matmul(Binv, YtSpSinvSqY.T)
                trace_SpMSqM_21 = numpy.trace(C21)
                trace_SpMSqM_22 = numpy.trace(C22)

                # Compute the third part of trace of Sp * M * Sq * M
                YtSpY = numpy.matmul(Y.T, SpY)
                if p == q:
                    YtSqY = YtSpY
                else:
                    YtSqY = numpy.matmul(Y.T, SqY)
                Dp = numpy.matmul(Binv, YtSpY)
                if p == q:
                    Dq = Dp
                else:
                    Dq = numpy.matmul(Binv, YtSqY)
                D = numpy.matmul(Dp, Dq)
                trace_SpMSqM_3 = numpy.trace(D)

                # Compute trace of Sp * M * Sq * M
                trace_SpMSqM = trace_SpMSqM_1 - trace_SpMSqM_21 - \
                    trace_SpMSqM_22 + trace_SpMSqM_3

                # 5. Second "local" derivatives w.r.t scale
                local_der2_scale = -0.5*trace_SpqM + 0.5*trace_SpMSqM - \
                    zMSqMSpMz + 0.5*zMSpqMz

                # Computing total second derivative
                MSqMz = M_dot(self.cov, Binv, Y, sigma, sigma0, SqMz)

                dp_log_sigma2 = -numpy.dot(self.z, MSpMz) / (n-m)
                dq_log_sigma2 = -numpy.dot(self.z, MSqMz) / (n-m)
                der2_scale[p, q] = local_der2_scale + \
                    0.5 * (n-m) * dp_log_sigma2 * dq_log_sigma2

                if p != q:
                    der2_scale[q, p] = der2_scale[p, q]

        return der2_scale

    # =====================
    # likelihood der2 mixed
    # =====================

    def _likelihood_der2_mixed(self, hyperparam):
        """
        lp is the log likelihood probability. der2_mixed is the mixed second
        derivative w.r.t eta and scale. The output is a 1D vector of the size
        of scale.
        """

        # Get log_eta
        if numpy.isscalar(hyperparam):
            log_eta = hyperparam
        else:
            log_eta = hyperparam[0]
        eta = 10.0**log_eta
        scale = numpy.abs(hyperparam[1:])
        self.cov.set_scale(scale)

        # Initialize mixed derivative as 2D array with one row.
        der2_mixed = numpy.zeros((1, scale.size), dtype=float)

        # Find optimal sigma based on eta. Then compute sigma0
        sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)

        n, m = self.X.shape

        # Computing Y=Sinv*X and w=Sinv*z.
        Y = self.cov.solve(sigma, sigma0, self.X)
        YtY = numpy.matmul(Y.T, Y)
        V = self.cov.solve(sigma, sigma0, Y)

        # B is Xt * Y
        B = numpy.matmul(self.X.T, Y)
        Binv = numpy.linalg.inv(B)

        # Compute Mz and MMz
        Mz = M_dot(self.cov, Binv, Y, sigma, sigma0, self.z)
        MMz = M_dot(self.cov, Binv, Y, sigma, sigma0, Mz)
        zMMz = numpy.dot(Mz, Mz)

        # Needed to compute trace (TODO)
        S = self.cov.get_matrix(sigma, sigma0)
        Sinv = numpy.linalg.inv(S)
        Sinv2 = Sinv @ Sinv

        # Sp is the derivative of cov w.r.t the p-th element of scale.
        for p in range(scale.size):

            # Compute zMSpMMz
            SpMz = self.cov.dot(sigma, sigma0, Mz, derivative=[p])
            zMSpMz = numpy.dot(Mz, SpMz)
            zMSpMMz = numpy.dot(SpMz, MMz)

            # Compute trace of SpSinv2
            Sp = self.cov.get_matrix(sigma, sigma0, derivative=[p])
            SpSinv2 = Sp @ Sinv2
            trace_SpSinv2, _ = imate.trace(SpSinv2, method='exact')

            # Compute traces
            SpY = self.cov.dot(sigma, sigma0, Y, derivative=[p])
            YtSpY = numpy.matmul(Y.T, SpY)
            VtSpY = numpy.matmul(V.T, SpY)
            C1 = numpy.matmul(Binv, VtSpY)
            C2 = numpy.matmul(Binv, VtSpY.T)
            D1 = numpy.matmul(Binv, YtSpY)
            D2 = numpy.matmul(Binv, YtY)
            D = numpy.matmul(D1, D2)

            trace_C1 = numpy.trace(C1)
            trace_C2 = numpy.trace(C2)
            trace_D = numpy.trace(D)

            # Compute trace of M * Sp * M
            trace_MSpM = trace_SpSinv2 - trace_C1 - trace_C2 + trace_D

            # Compute mixed derivative
            der2_mixed[p] = sigma**2 * (0.5*trace_MSpM - zMSpMMz +
                                        (0.5/(n-m)) * zMMz * zMSpMz)

        return der2_mixed

    # ===================
    # likelihood jacobian
    # ===================

    def likelihood_jacobian(self, sign_switch, hyperparam):
        """
        Computes Jacobian w.r.t eta, and if given, scale.
        """

        # Derivative w.r.t eta
        der1_eta = self._likelihood_der1_eta(hyperparam)

        jacobian = der1_eta

        # Compute Jacobian w.r.t scale
        if hyperparam.size > 1:

            # Compute first derivative w.r.t scale
            der1_scale = self._likelihood_der1_scale(hyperparam)

            # Concatenate derivatives of eta and scale if needed
            jacobian = numpy.r_[jacobian, der1_scale]

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

        der2_eta = self._likelihood_der2_eta(hyperparam)

        # hessian here is a 2D array of size 1
        hessian = der2_eta

        # Compute Hessian w.r.t scale
        if hyperparam.size > 1:

            # Compute second derivative w.r.t scale
            der2_scale = self._likelihood_der2_scale(hyperparam)

            # Compute second mixed derivative w.r.t scale and eta
            der2_mixed = self._likelihood_der2_mixed(hyperparam)

            # Concatenate derivatives to form Hessian of all variables
            hessian = numpy.block(
                    [[hessian, der2_mixed],
                     [der2_mixed.T, der2_scale]])

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

        max_eta = 1e+16
        min_eta = 1e-16
        if numpy.abs(eta) > max_eta:

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
            if numpy.abs(eta) < min_eta:
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
        root finding of the derivative of lp.

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
            # There is a sign change in the interval of eta. Find root of lp
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
            # eta = 10**res.root                       # Use with Brent
            eta = 10**res['root']                      # Use with Chandrupatla
            sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)
            iter = res['iterations']

            # Check second derivative
            # success = True
            # d2lp_deta2 = self._likelihood_der2_eta(eta)
            # if d2lp_deta2 < 0:
            #     success = True
            # else:
            #     success = False

        else:
            # bracket with sign change was not found.
            iter = 0

            # Evaluate the function in intervals
            eta_left = bracket[0]
            eta_right = bracket[1]
            dlp_deta_left = bracket_values[0]
            dlp_deta_right = bracket_values[1]

            # Second derivative of log likelihood at eta = zero, using either
            # of the two methods below:
            eta_zero = 0.0
            # method 1: directly from analytical equation
            d2lp_deta2_zero_eta = self._likelihood_der2_eta(eta_zero)

            # method 2: using forward differencing from first derivative
            # dlp_deta_zero_eta = self._likelihood_der1_eta(
            #         numpy.log10(eta_zero))
            # d2lp_deta2_zero_eta = \
            #         (dlp_deta_lowest_eta - dlp_deta_zero_eta) / eta_lowest

            # print('dL/deta   at eta = 0.0:\t %0.2f'%dlp_deta_zero_eta)
            print('dL/deta   at eta = %0.2e:\t %0.2f'
                  % (eta_left, dlp_deta_left))
            print('dL/deta   at eta = %0.2e:\t %0.16f'
                  % (eta_right, dlp_deta_right))
            print('d2L/deta2 at eta = 0.0:\t %0.2f'
                  % d2lp_deta2_zero_eta)

            # No sign change. Can not find a root
            if (dlp_deta_left > 0) and (dlp_deta_right > 0):
                if d2lp_deta2_zero_eta > 0:
                    eta = 0.0
                else:
                    eta = numpy.inf

            elif (dlp_deta_left < 0) and (dlp_deta_right < 0):
                if d2lp_deta2_zero_eta < 0:
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

        if optimization_method == 'chandrupatla':

            if len(hyperparam_guess) > 1:
                warnings.warn('"chandrupatla" method does not optimize ' +
                              '"scale". The "distance scale in the given ' +
                              '"hyperparam_guess" will be ignored. To ' +
                              'optimize distance scale with "chandrupatla"' +
                              'method, set "profile_eta" to True.')
                scale_guess = hyperparam_guess[1:]
                if self.cov.get_scale() is None:
                    self.cov.set_scale(scale_guess)
                    warnings.warn('"scale" is set based on the guess value.')

            # Note: When using interpolation, make sure the interval below is
            # exactly the end points of eta_i, not less or more.
            log_eta_guess = hyperparam_guess[0]
            min_eta_guess = numpy.min([1e-4, 10.0**log_eta_guess * 1e-2])
            max_eta_guess = numpy.max([1e+3, 10.0**log_eta_guess * 1e+2])
            interval_eta = [min_eta_guess, max_eta_guess]

            # Using root finding method on the first derivative w.r.t eta
            result = self._find_likelihood_der1_zeros(interval_eta)

            # Finding the maxima. This isn't necessary and affects run time
            result['optimization']['max_likelihood'] = self.likelihood(
                    False, result['hyperparam']['eta'])

            # The distance scale used in this method is the same as its guess.
            result['hyperparam']['scale'] = self.cov.get_scale()

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
            log_eta = res.x[0]
            if numpy.isneginf(log_eta):
                eta = 0.0
            else:
                eta = 10.0**log_eta
            # eta = log_eta   # Test
            sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)
            max_lp = -res.fun

            # Distance scale
            if res.x.size > 1:
                scale = numpy.abs(res.x[1:])
            else:
                scale = self.cov.get_scale()

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
                    'max_likelihood': max_lp,
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
        dimension = self.cov.mixed_cor.cor.dimension
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

        # Generate lp for various distance scales
        scale = numpy.logspace(-3, 2, 100)

        d0_lp = numpy.zeros((stencil_size, etas.size, scale.size),
                            dtype=float)
        d1_lp = numpy.zeros((etas.size, scale.size), dtype=float)
        d2_lp = numpy.zeros((etas.size, scale.size), dtype=float)
        d2_mixed_lp = numpy.zeros((etas.size, scale.size),
                                  dtype=float)
        d1_lp_numerical = numpy.zeros((stencil_size, etas.size, scale.size-2),
                                      dtype=float)
        d2_lp_numerical = numpy.zeros((etas.size, scale.size-4),
                                      dtype=float)
        d2_mixed_lp_numerical = numpy.zeros((etas.size, scale.size-2),
                                            dtype=float)

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        colors = matplotlib.cm.nipy_spectral(
                numpy.linspace(0, 0.9, etas.size))

        for i in range(etas.size):

            # Stencil to perturb eta
            d_eta = etas[i] * 1e-3
            eta_stencil = etas[i] + \
                d_eta * numpy.arange(-stencil_size//2+1, stencil_size//2+1)

            for j in range(scale.size):

                # Set the scale
                self.cov.set_scale(scale[j])

                # Likelihood (first index, center_stencil, means the main etas)
                for k in range(stencil_size):
                    d0_lp[k, i, j] = self.likelihood(
                            False, numpy.log10(eta_stencil[k]))

                # First derivative of likelihood w.r.t distance scale
                hyperparam = numpy.r_[numpy.log10(etas[i]), scale[j]]
                d1_lp[i, j] = self._likelihood_der1_scale(hyperparam)[0]

                # Second derivative of likelihood w.r.t distance scale
                d2_lp[i, j] = self._likelihood_der2_scale(hyperparam)[0, 0]

                # Second mixed derivative of likelihood w.r.t distance scale
                d2_mixed_lp[i, j] = self._likelihood_der2_mixed(hyperparam)[0]

            for k in range(stencil_size):
                # Compute first derivative numerically
                d1_lp_numerical[k, i, :] = \
                        (d0_lp[k, i, 2:] - d0_lp[k, i, :-2]) / \
                        (scale[2:] - scale[:-2])

                # Second mixed derivative numerically (finite difference)
                d2_mixed_lp_numerical[i, :] += \
                    coeff[k] * d1_lp_numerical[k, i, :] / d_eta

            # Compute second derivative numerically
            d2_lp_numerical[i, :] = \
                (d1_lp_numerical[center_stencil, i, 2:] -
                 d1_lp_numerical[center_stencil, i, :-2]) / \
                (scale[3:-1] - scale[1:-3])

            # Find maximum of lp
            max_index = numpy.argmax(d0_lp[center_stencil, i, :])
            optimal_scale = scale[max_index]
            optimal_lp = d0_lp[center_stencil, i, max_index]

            # Plot
            if etas[i] == optimal_eta:
                label = r'$\hat{\eta}=%0.2e$' % etas[i]
                marker = 'X'
            else:
                label = r'$\eta=%0.2e$' % etas[i]
                marker = 'o'
            ax[0, 0].plot(scale, d0_lp[center_stencil, i, :], color=colors[i],
                          label=label)
            ax[0, 1].plot(scale, d1_lp[i, :], color=colors[i], label=label)
            ax[1, 0].plot(scale, d2_lp[i, :], color=colors[i], label=label)
            ax[1, 1].plot(scale, d2_mixed_lp[i, :], color=colors[i],
                          label=label)
            ax[0, 1].plot(scale[1:-1], d1_lp_numerical[center_stencil, i, :],
                          '--', color=colors[i])
            ax[1, 0].plot(scale[2:-2], d2_lp_numerical[i, :], '--',
                          color=colors[i])
            ax[1, 1].plot(scale[1:-1], d2_mixed_lp_numerical[i, :], '--',
                          color=colors[i])
            p = ax[0, 0].plot(optimal_scale, optimal_lp, marker,
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
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d} \ell(\theta | \eta)}{\mathrm{d} \theta}$')
        ax[1, 0].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} \theta^2}$')
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} \theta ' +
            r'\mathrm{d} \eta}$')
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
        dimension = self.cov.mixed_cor.cor.dimension
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
        d0_lp = numpy.zeros((stencil_size, scales.size, eta.size,),
                            dtype=float)
        d1_lp = numpy.zeros((scales.size, eta.size,), dtype=float)
        d2_lp = numpy.zeros((scales.size, eta.size,), dtype=float)
        d2_mixed_lp = numpy.zeros((scales.size, eta.size), dtype=float)
        d1_lp_numerical = numpy.zeros((stencil_size, scales.size, eta.size-2,),
                                      dtype=float)
        d2_lp_numerical = numpy.zeros((scales.size, eta.size-4,), dtype=float)
        d2_mixed_lp_numerical = numpy.zeros((scales.size, eta.size-2),
                                            dtype=float)

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        colors = matplotlib.cm.nipy_spectral(
                numpy.linspace(0, 0.9, scales.size))

        for i in range(scales.size):

            # Stencil to perturb scale
            d_scale = scales[i] * 1e-3
            scale_stencil = scales[i] + \
                d_scale * numpy.arange(-stencil_size//2+1, stencil_size//2+1)

            # Iterate over the perturbations of scale
            for k in range(stencil_size):

                # Set the perturbed distance scale
                self.cov.set_scale(scale_stencil[k])

                for j in range(eta.size):

                    # Likelihood
                    d0_lp[k, i, j] = self.likelihood(
                            False, numpy.log10(eta[j]))

                    if k == center_stencil:
                        # First derivative w.r.t eta
                        d1_lp[i, j] = self._likelihood_der1_eta(
                                numpy.log10(eta[j]))

                        # Second derivative w.r.t eta
                        d2_lp[i, j] = self._likelihood_der2_eta(
                                numpy.log10(eta[j]))

                        # Second mixed derivative w.r.t distance scale and eta
                        hyperparam = numpy.r_[numpy.log10(eta[j]),
                                              scale_stencil[k]]
                        d2_mixed_lp[i, j] = self._likelihood_der2_mixed(
                                hyperparam)[0]

            for k in range(stencil_size):
                # Compute first derivative numerically
                d1_lp_numerical[k, i, :] = \
                    (d0_lp[k, i, 2:] - d0_lp[k, i, :-2]) / \
                    (eta[2:] - eta[:-2])

                # Second mixed derivative numerically (finite difference)
                d2_mixed_lp_numerical[i, :] += \
                    coeff[k] * d1_lp_numerical[k, i, :] / d_scale

            # Compute second derivative numerically
            d2_lp_numerical[i, :] = \
                (d1_lp_numerical[center_stencil, i, 2:] -
                 d1_lp_numerical[center_stencil, i, :-2]) / \
                (eta[3:-1] - eta[1:-3])

            # Find maximum of lp
            max_index = numpy.argmax(d0_lp[center_stencil, i, :])
            optimal_eta = eta[max_index]
            optimal_lp = d0_lp[center_stencil, i, max_index]

            if scales[i] == optimal_scale:
                label = r'$\hat{\theta} = %0.2e$' % scales[i]
                marker = 'X'
            else:
                label = r'$\theta = %0.2e$' % scales[i]
                marker = 'o'

            ax[0, 0].plot(eta, d0_lp[center_stencil, i, :], color=colors[i],
                          label=label)
            ax[0, 1].plot(eta, d1_lp[i, :], color=colors[i], label=label)
            ax[1, 0].plot(eta, d2_lp[i, :], color=colors[i], label=label)
            ax[1, 1].plot(eta, d2_mixed_lp[i, :], color=colors[i], label=label)
            ax[0, 1].plot(eta[1:-1], d1_lp_numerical[center_stencil, i, :],
                          '--', color=colors[i])
            ax[1, 0].plot(eta[2:-2], d2_lp_numerical[i, :], '--',
                          color=colors[i])
            ax[1, 1].plot(eta[1:-1], d2_mixed_lp_numerical[i, :], '--',
                          color=colors[i])
            p = ax[0, 0].plot(optimal_eta, optimal_lp, marker,
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
        ax[0, 0].set_xlabel(r'$\eta$')
        ax[0, 1].set_xlabel(r'$\eta$')
        ax[1, 0].set_xlabel(r'$\eta$')
        ax[1, 1].set_xlabel(r'$\eta$')
        ax[0, 0].set_ylabel(r'$\ell(\eta | \theta)$')
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d}\ell(\eta | \theta)}{\mathrm{d}\eta}$')
        ax[1, 0].set_ylabel(
            r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d}\eta^2}$')
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
        dimension = self.cov.mixed_cor.cor.dimension
        if dimension != 1:
            raise ValueError('To plot likelihood w.r.t "eta" and "scale", ' +
                             'the dimension of the data points should be one.')

        load_plot_settings()

        # Optimal point
        optimal_eta = result['hyperparam']['eta']
        optimal_scale = result['hyperparam']['scale']
        optimal_lp = result['optimization']['max_likelihood']

        eta = numpy.logspace(-3, 3, 50)
        scale = numpy.logspace(-3, 2, 50)
        lp = numpy.zeros((scale.size, eta.size), dtype=float)

        # Compute lp
        for i in range(scale.size):
            self.cov.set_scale(scale[i])
            for j in range(eta.size):
                lp[i, j] = self.likelihood(False, numpy.log10(eta[j]))

        # Convert inf to nan
        lp = numpy.where(numpy.isinf(lp), numpy.nan, lp)

        # Smooth data for finer plot
        # sigma_ = [2, 2]  # in unit of data pixel size
        # lp = scipy.ndimage.filters.gaussian_filter(
        #         lp, sigma_, mode='nearest')

        # Increase resolution for better contour plot
        N = 300
        f = scipy.interpolate.interp2d(
                numpy.log10(eta), numpy.log10(scale), lp, kind='cubic')
        scale_fine = numpy.logspace(numpy.log10(scale[0]),
                                    numpy.log10(scale[-1]), N)
        eta_fine = numpy.logspace(numpy.log10(eta[0]), numpy.log10(eta[-1]), N)
        x, y = numpy.meshgrid(eta_fine, scale_fine)
        lp_fine = f(numpy.log10(eta_fine), numpy.log10(scale_fine))

        # We will plot the difference of max of Lp to Lp, called z
        # max_lp = numpy.abs(numpy.max(lp_fine))
        # z = max_lp - lp_fine
        z = lp_fine

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
        opt_lp1 = numpy.zeros((eta_fine.size, ), dtype=float)
        opt_lp1[:] = numpy.nan
        for j in range(eta_fine.size):
            if numpy.all(numpy.isnan(lp_fine[:, j])):
                continue
            max_index = numpy.nanargmax(lp_fine[:, j])
            opt_scale1[j] = scale_fine[max_index]
            opt_lp1[j] = lp_fine[max_index, j]
        ax[0].plot(eta_fine, opt_scale1, color='red',
                   label=r'$\hat{\theta}(\eta)$')
        ax[1].plot(eta_fine, opt_lp1, color='red')

        # Find max for each fixed scale
        opt_eta2 = numpy.zeros((scale_fine.size, ), dtype=float)
        opt_lp2 = numpy.zeros((scale_fine.size, ), dtype=float)
        opt_lp2[:] = numpy.nan
        for i in range(scale_fine.size):
            if numpy.all(numpy.isnan(lp_fine[i, :])):
                continue
            max_index = numpy.nanargmax(lp_fine[i, :])
            opt_eta2[i] = eta_fine[max_index]
            opt_lp2[i] = lp_fine[i, max_index]
        ax[0].plot(opt_eta2, scale_fine, color='black',
                   label=r'$\hat{\eta}(\theta)$')
        ax[2].plot(scale_fine, opt_lp2, color='black')

        # Plot max of the whole 2D array
        max_indices = numpy.unravel_index(numpy.nanargmax(lp_fine),
                                          lp_fine.shape)
        opt_scale = scale_fine[max_indices[0]]
        opt_eta = eta_fine[max_indices[1]]
        opt_lp = lp_fine[max_indices[0], max_indices[1]]
        ax[0].plot(opt_eta, opt_scale, 'o', color='red', markersize=6,
                   label=r'$(\hat{\eta}, \hat{\theta})$ (by brute force on ' +
                         r'grid)')
        ax[1].plot(opt_eta, opt_lp, 'o', color='red',
                   label=r'$\ell(\hat{\eta}, \hat{\theta})$ (by brute force ' +
                         r'on grid)')
        ax[2].plot(opt_scale, opt_lp, 'o', color='red',
                   label=r'$\ell(\hat{\eta}, \hat{\theta})$ (by brute force ' +
                         r'on grid)')

        # Plot optimal point as found by the profile likelihood method
        ax[0].plot(optimal_eta, optimal_scale, 'o', color='black',
                   markersize=6,
                   label=r'$\max_{\eta, \theta} \ell$ (by optimization)')
        ax[1].plot(optimal_eta, optimal_lp, 'o', color='black',
                   label=r'$\ell(\hat{\eta}, \hat{\theta})$ (by optimization)')
        ax[2].plot(optimal_scale, optimal_lp, 'o', color='black',
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
        dlp_deta_upper_bound = 0.5*(n-m) * \
            (1/(eta+eigenvalue_smallest) - 1/(eta+eigenvalue_largest))
        dlp_deta_lower_bound = -dlp_deta_upper_bound

        return dlp_deta_upper_bound, dlp_deta_lower_bound

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

        print('Plot first derivative ...')

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
        dlp_deta = numpy.zeros(eta.size)
        for i in range(eta.size):
            dlp_deta[i] = self._likelihood_der1_eta(numpy.log10(eta[i]))

        # Compute upper and lower bound of derivative
        K = self.mixed_cor.get_matrix(0.0)
        dlp_deta_upper_bound, dlp_deta_lower_bound = \
            self._compute_bounds_der1_eta(K, eta)

        # Compute asymptote of first derivative, using both first and second
        # order approximation
        try:
            # eta_high_res might not be defined, depending on plot_optimal_eta
            x = eta_high_res
        except NameError:
            x = numpy.logspace(1, log_eta_end, 100)
        dlp_deta_asymptote_1, dlp_deta_asymptote_2, roots_1, roots_2 = \
            self._compute_asymptote_der1_eta(K, x)

        # Main plot
        fig, ax1 = plt.subplots()
        ax1.semilogx(eta, dlp_deta_upper_bound, '--', color='black',
                     label='Upper bound')
        ax1.semilogx(eta, dlp_deta_lower_bound, '-.', color='black',
                     label='Lower bound')
        ax1.semilogx(eta, dlp_deta, color='black', label='Exact')
        if plot_optimal_eta:
            ax1.semilogx(optimal_eta, 0, '.', marker='o', markersize=4,
                         color='black')

        # Min of plot limit
        # ax1.set_yticks(numpy.r_[numpy.arange(-120, 1, 40), 20])
        max_plot = numpy.max(dlp_deta)
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

            ax2.semilogx(eta, numpy.abs(dlp_deta_upper_bound), '--',
                         color='black')
            ax2.semilogx(eta, numpy.abs(dlp_deta_lower_bound), '-.',
                         color='black')
            ax2.semilogx(x, dlp_deta_asymptote_1,
                         label=r'$1^{\text{st}}$ order asymptote',
                         color='chocolate')
            ax2.semilogx(x, dlp_deta_asymptote_2,
                         label=r'$2^{\text{nd}}$ order asymptote',
                         color='olivedrab')
            ax2.semilogx(eta_high_res,
                         dlp_deta[eta_low_res_left.size:
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
            min_plot = numpy.abs(numpy.min(dlp_deta))
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
