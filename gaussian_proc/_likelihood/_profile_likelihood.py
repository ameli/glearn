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
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.optimize
from functools import partial

from .._utilities.plot_utilities import *                    # noqa: F401, F403
from .._utilities.plot_utilities import load_plot_settings, save_plot, plt, \
        mark_inset, InsetPosition, matplotlib
from ._root_finding import find_interval_with_sign_change, chandrupatla_method
from ._likelihood_utilities import M_dot
import imate
import warnings


# ==================
# Profile Likelihood
# ==================

class ProfileLikelihood(object):

    # ==========
    # Likelihood
    # ==========

    def likelihood(z, X, mixed_cor, sign_switch, hyperparam):
        """
        Log likelihood function

            L = -(1/2) log det(S) - (1/2) log det(X.T*Sinv*X) -
                (1/2) sigma^(-2) * z.T * M1 * z

        where
            S = sigma^2 Kn is the covariance
            Sinv is the inverse of S
            M1 = Sinv = Sinv*X*(X.T*Sinv*X)^(-1)*X.T*Sinv

        hyperparam = [eta, distance_scale[0], distance_scale[1], ...]

        sign_switch changes the sign of the output from lp to -lp. When True,
        this is used to minimizing (instead of maximizing) the negative of
        log-likelihood function.
        """

        if (not numpy.isscalar(hyperparam)) and (hyperparam.size > 1):
            distance_scale = numpy.abs(hyperparam[1:])
            mixed_cor.set_distance_scale(distance_scale)

            # Test
            # print(distance_scale)
            # if any(distance_scale > 1.0):
            #     return 0.0

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

        n, m = X.shape

        max_eta = 1e+16
        if numpy.abs(eta) >= max_eta:

            B = numpy.matmul(X.T, X)
            Binv = numpy.linalg.inv(B)
            logdet_Binv = numpy.log(numpy.linalg.det(Binv))

            # Optimal sigma0 when eta is very large
            sigma0 = ProfileLikelihood.find_optimal_sigma0(z, X)

            # Log likelihood
            lp = -0.5*(n-m)*numpy.log(2.0*numpy.pi) \
                - (n-m)*numpy.log(sigma0) - 0.5*logdet_Binv - 0.5*(n-m)

        else:

            sigma = ProfileLikelihood.find_optimal_sigma(z, X, mixed_cor, eta)
            logdet_Kn = mixed_cor.logdet(eta)

            # Compute log det (X.T Kn_inv X)
            Y = mixed_cor.solve(eta, X)
            w = mixed_cor.solve(eta, z)

            XtKninvX = numpy.matmul(X.T, Y)
            logdet_XtKninvX = numpy.log(numpy.linalg.det(XtKninvX))

            # Suppose B is XtKninvX found above. We compute inverse of B
            Binv = numpy.linalg.inv(XtKninvX)
            YBinvYt = numpy.matmul(Y, numpy.matmul(Binv, Y.T))

            # Log likelihood
            lp = -0.5*(n-m)*numpy.log(2.0*numpy.pi) \
                - (n-m)*numpy.log(sigma) - 0.5*logdet_Kn \
                - 0.5*logdet_XtKninvX \
                - (0.5/(sigma**2))*numpy.dot(z, w-numpy.dot(YBinvYt, z))

        # If lp is used in scipy.optimize.minimize, change the sign to obtain
        # the minimum of -lp
        if sign_switch:
            lp = -lp

        return lp

    # ===================
    # likelihood der1 eta
    # ===================

    def likelihood_der1_eta(z, X, mixed_cor, hyperparam):
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

        # Include derivative w.r.t distance_scale
        if (not numpy.isscalar(hyperparam)) and (hyperparam.size > 1):
            distance_scale = numpy.abs(hyperparam[1:])
            mixed_cor.set_distance_scale(distance_scale)

        # Compute Kn_inv*X and Kn_inv*z
        Y = mixed_cor.solve(eta, X)
        w = mixed_cor.solve(eta, z)

        n, m = X.shape

        # Splitting M into M1 and M2. Here, we compute M2
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)
        Ytz = numpy.matmul(Y.T, z)
        Binv_Ytz = numpy.matmul(Binv, Ytz)
        Y_Binv_Ytz = numpy.matmul(Y, Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        # Traces
        trace_Kninv = mixed_cor.traceinv(eta)
        YtY = numpy.matmul(Y.T, Y)
        trace_BinvYtY = numpy.trace(numpy.matmul(Binv, YtY))
        trace_M = trace_Kninv - trace_BinvYtY

        # Derivative of log likelihood
        zMz = numpy.dot(z, Mz)
        zM2z = numpy.dot(Mz, Mz)
        sigma2 = zMz/(n-m)
        dlp_deta = -0.5*(trace_M - zM2z/sigma2)

        # Because we use xi = log_eta instead of eta as the variable, the
        # derivative of lp w.r.t log_eta is dlp_deta * deta_dxi, and
        # deta_dxi is eta * lob(10).
        # dlp_deta = dlp_deta * eta * numpy.log(10.0)

        # Return as scalar or array of length one
        if numpy.isscalar(hyperparam):
            return dlp_deta
        else:
            return numpy.array([dlp_deta], dtype=float)

    # ===================
    # likelihood der2 eta
    # ===================

    @staticmethod
    def likelihood_der2_eta(z, X, mixed_cor, hyperparam):
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

        # Include derivative w.r.t distance_scale
        if (not numpy.isscalar(hyperparam)) and (hyperparam.size > 1):
            distance_scale = numpy.abs(hyperparam[1:])
            mixed_cor.set_distance_scale(distance_scale)

        Y = mixed_cor.solve(eta, X)
        V = mixed_cor.solve(eta, Y)
        w = mixed_cor.solve(eta, z)

        n, m = X.shape

        # Splitting M
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)
        Ytz = numpy.matmul(Y.T, z)
        Binv_Ytz = numpy.matmul(Binv, Ytz)
        Y_Binv_Ytz = numpy.matmul(Y, Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        # Trace of M
        # trace_Kninv = mixed_cor.traceinv(eta)
        YtY = numpy.matmul(Y.T, Y)
        A = numpy.matmul(Binv, YtY)
        # trace_A = numpy.trace(A)
        # trace_M = trace_Kninv - trace_A

        # Trace of M**2
        trace_Kn2inv = mixed_cor.traceinv(eta, exponent=2)
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
        v = mixed_cor.solve(eta, Mz)
        MMz = v - Y_Binv_YtMz

        # Second derivative (only at the location of zero first derivative)
        zMz = numpy.dot(z, Mz)
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
            return numpy.array([d2lp_deta2])

    # ==============================
    # likelihood der1 distance scale
    # ==============================

    def likelihood_der1_distance_scale(z, X, cov, hyperparam):
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
        distance_scale = numpy.abs(hyperparam[1:])
        cov.set_distance_scale(distance_scale)

        # Initialize jacobian
        der1_distance_scale = numpy.zeros((distance_scale.size, ),
                                          dtype=float)

        # Find optimal sigma based on eta. Then compute sigma0
        sigma, sigma0 = ProfileLikelihood.find_optimal_sigma_sigma0(
                z, X, cov.mixed_cor, eta)

        n, m = X.shape

        # Computing Y=Sinv*X and w=Sinv*z.
        Y = cov.solve(sigma, sigma0, X)

        # B is Xt * Y
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)

        # Compute Mz
        Mz = M_dot(cov, Binv, Y, sigma, sigma0, z)

        # Needed to compute trace (TODO)
        S = cov.get_matrix(sigma, sigma0)
        Sinv = numpy.linalg.inv(S)

        # Sp is the derivative of cov w.r.t the p-th element of
        # distance_scale.
        for p in range(distance_scale.size):

            # Compute zMSpMz
            SpMz = cov.dot(sigma, sigma0, Mz, derivative=[p])
            zMSpMz = numpy.dot(Mz, SpMz)

            # Compute the first component of trace of Sp * M (TODO)
            Sp = cov.get_matrix(sigma, sigma0, derivative=[p])
            SpSinv = Sp @ Sinv
            trace_SpSinv, _ = imate.trace(SpSinv, method='exact')
            # trace_SpSinv = cov.traceinv(sigma, sigma0, Sp,
            #                             imate_method='hutchinson')

            # Compute the second component of trace of Sp * M
            SpY = cov.dot(sigma, sigma0, Y, derivative=[p])
            YtSpY = numpy.matmul(Y.T, SpY)
            BinvYtSpY = numpy.matmul(Binv, YtSpY)
            trace_BinvYtSpY = numpy.trace(BinvYtSpY)

            # Compute trace of Sp * M
            trace_SpM = trace_SpSinv - trace_BinvYtSpY

            # Derivative of lp w.r.t p-th element of distance scale
            der1_distance_scale[p] = -0.5*trace_SpM + 0.5*zMSpMz

            # Test
            # if distance_scale[p] > 1.0:
            #     der1_distance_scale[p] = 0.0

        return der1_distance_scale

    # ==============================
    # likelihood der2 distance scale
    # ==============================

    def likelihood_der2_distance_scale(z, X, cov, hyperparam):
        """
        lp is the log likelihood probability. der2_distance_scale is
        d2(lp)/d(theta2), is the second derivative of lp with respect to the
        distance scale (theta).
        """

        # Get log_eta
        if numpy.isscalar(hyperparam):
            log_eta = hyperparam
        else:
            log_eta = hyperparam[0]
        eta = 10.0**log_eta
        distance_scale = numpy.abs(hyperparam[1:])
        cov.set_distance_scale(distance_scale)

        # Initialize hessian
        der2_distance_scale = numpy.zeros(
                (distance_scale.size, distance_scale.size), dtype=float)

        # Find optimal sigma based on eta. Then compute sigma0
        sigma, sigma0 = ProfileLikelihood.find_optimal_sigma_sigma0(
                z, X, cov.mixed_cor, eta)

        n, m = X.shape

        # Computing Y=Sinv*X and w=Sinv*z.
        Y = cov.solve(sigma, sigma0, X)

        # B is Xt * Y
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)

        # Compute Mz
        Mz = M_dot(cov, Binv, Y, sigma, sigma0, z)

        # Needed to compute trace (TODO)
        S = cov.get_matrix(sigma, sigma0)
        Sinv = numpy.linalg.inv(S)

        # Test
        # M = Sinv - Y @ Binv @ Y.T

        # Sp is the derivative of cov w.r.t the p-th element of
        # distance_scale.
        for p in range(distance_scale.size):

            SpMz = cov.dot(sigma, sigma0, Mz, derivative=[p])
            MSpMz = M_dot(cov, Binv, Y, sigma, sigma0, SpMz)

            for q in range(distance_scale.size):

                # # Test
                # Sp = cov.get_matrix(sigma, sigma0, derivative=[p])
                # Sq = cov.get_matrix(sigma, sigma0, derivative=[q])
                # Spq = cov.get_matrix(sigma, sigma0, derivative=[p, q])
                # Q = Spq @ M - Sp @ M @ Sq @ M
                # W = M @ Spq @ M - 2.0 * M @ Sp @ M @ Sq @ M
                # test = -0.5*numpy.trace(Q) + \
                #         0.5*numpy.dot(z, W @ z)

                # 1. Compute zMSqMSpMz
                if p == q:
                    SqMz = SpMz
                else:
                    SqMz = cov.dot(sigma, sigma0, Mz, derivative=[q])
                zMSqMSpMz = numpy.dot(SqMz, MSpMz)

                # 2. Compute zMSpqMz
                SpqMz = cov.dot(sigma, sigma0, Mz, derivative=[p, q])
                zMSpqMz = numpy.dot(Mz, SpqMz)

                # 3. Computing trace of Spq * M in three steps

                # Compute the first component of trace of Spq * Sinv (TODO)
                Spq = cov.get_matrix(sigma, sigma0, derivative=[p, q])
                SpqSinv = Spq @ Sinv
                trace_SpqSinv, _ = imate.trace(SpqSinv, method='exact')

                # Compute the second component of trace of Spq * M
                SpqY = cov.dot(sigma, sigma0, Y, derivative=[p, q])
                YtSpqY = numpy.matmul(Y.T, SpqY)
                BinvYtSpqY = numpy.matmul(Binv, YtSpqY)
                trace_BinvYtSpqY = numpy.trace(BinvYtSpqY)

                # Compute trace of Spq * M
                trace_SpqM = trace_SpqSinv - trace_BinvYtSpqY

                # 4. Compute trace of Sp * M * Sq * M

                # Compute first part of trace of Sp * M * Sq * M
                Sp = cov.get_matrix(sigma, sigma0, derivative=[p])
                SpSinv = Sp @ Sinv
                Sq = cov.get_matrix(sigma, sigma0, derivative=[q])
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
                SinvSqY = cov.solve(sigma, sigma0, SqY)
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

                # 5. Second derivatives w.r.t distance_scale
                der2_distance_scale[p, q] = -0.5*trace_SpqM + \
                    0.5*trace_SpMSqM - zMSqMSpMz + 0.5*zMSpqMz

                if p != q:
                    der2_distance_scale[q, p] = der2_distance_scale[p, q]

        return der2_distance_scale

    # ===================
    # likelihood jacobian
    # ===================

    @staticmethod
    def likelihood_jacobian(z, X, cov, sign_switch, hyperparam):
        """
        Computes Jacobian w.r.t eta, and if given, distance_scale.
        """

        # Derivative w.r.t eta
        der1_eta = ProfileLikelihood.likelihood_der1_eta(
                z, X, cov.mixed_cor, hyperparam)

        jacobian = der1_eta

        # Compute Jacobian w.r.t distance_scale
        if hyperparam.size > 1:

            # Compute first derivative w.r.t distance_scale
            der1_distance_scale = \
                    ProfileLikelihood.likelihood_der1_distance_scale(
                            z, X, cov, hyperparam)

            # Concatenate derivatives of eta and distance_scale if needed
            jacobian = numpy.r_[jacobian, der1_distance_scale]

        if sign_switch:
            jacobian = -jacobian

        return jacobian

    # =========================
    # find optimal sigma sigma0
    # =========================

    @staticmethod
    def find_optimal_sigma_sigma0(z, X, mixed_cor, eta):
        """
        Based on a given eta, finds optimal sigma and sigma0.
        """

        max_eta = 1e+16
        min_eta = 1e-16
        if numpy.abs(eta) > max_eta:

            # eta is very large. Use Asymptotic relation
            sigma0 = ProfileLikelihood.find_optimal_sigma0(z, X)

            if numpy.isinf(eta):
                sigma = 0.
            else:
                sigma = sigma0 / numpy.sqrt(eta)

        else:

            # Find sigma
            sigma = ProfileLikelihood.find_optimal_sigma(z, X, mixed_cor, eta)

            # Find sigma0
            if numpy.abs(eta) < min_eta:
                sigma0 = 0.0
            else:
                sigma0 = numpy.sqrt(eta) * sigma

        return sigma, sigma0

    # ==================
    # find optimal sigma
    # ==================

    @staticmethod
    def find_optimal_sigma(z, X, mixed_cor, eta):
        """
        When eta is *not* very large, finds optimal sigma.
        """

        Y = mixed_cor.solve(eta, X)
        w = mixed_cor.solve(eta, z)

        n, m = X.shape
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)
        Ytz = numpy.matmul(Y.T, z)
        v = numpy.matmul(Y, numpy.matmul(Binv, Ytz))
        sigma2 = numpy.dot(z, w-v) / (n-m)
        sigma = numpy.sqrt(sigma2)

        return sigma

    # ===================
    # find optimal sigma0
    # ===================

    @staticmethod
    def find_optimal_sigma0(z, X):
        """
        When eta is very large, we assume sigma is zero. Thus, sigma0 is
        computed by this function.
        """

        n, m = X.shape
        B = numpy.matmul(X.T, X)
        Binv = numpy.linalg.inv(B)
        Xtz = numpy.matmul(X.T, z)
        v = numpy.matmul(X, numpy.matmul(Binv, Xtz))
        sigma02 = numpy.dot(z, z-v) / (n-m)
        sigma0 = numpy.sqrt(sigma02)

        return sigma0

    # ==========================
    # find likelihood der1 zeros
    # ==========================

    def find_likelihood_der1_zeros(z, X, mixed_cor, interval_eta, tol=1e-6,
                                   max_iterations=100, num_bracket_trials=3):
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

        # Partial function with minus to make maximization to a minimization
        likelihood_der1_eta_partial_function = partial(
                ProfileLikelihood.likelihood_der1_eta, z, X, mixed_cor)

        # Initial points
        bracket = [log_eta_start, log_eta_end]
        bracket_found, bracket, bracket_values = \
            find_interval_with_sign_change(
                    likelihood_der1_eta_partial_function, bracket,
                    num_bracket_trials, args=(), )

        if bracket_found:
            # There is a sign change in the interval of eta. Find root of lp
            # derivative

            # Find roots using Brent method
            # method = 'brentq'
            # res = scipy.optimize.root_scalar(
            #         likelihood_der1_eta_partial_function,
            #         bracket=bracket,
            #         method=method, xtol=tol)
            # print('Iter: %d, Eval: %d, Converged: %s'
            #         % (res.iterations, res.function_calls, res.converged))

            # Find roots using Chandraputala method
            res = chandrupatla_method(likelihood_der1_eta_partial_function,
                                      bracket, bracket_values, verbose=False,
                                      eps_m=tol, eps_a=tol,
                                      maxiter=max_iterations)

            # Extract results
            # eta = 10**res.root                       # Use with Brent
            eta = 10**res['root']                      # Use with Chandrupatla
            sigma, sigma0 = ProfileLikelihood.find_optimal_sigma_sigma0(
                    z, X, mixed_cor, eta)
            iter = res['iterations']

            # Check second derivative
            # success = True
            # d2lp_deta2 = ProfileLikelihood.likelihood_der2_eta(
            #         z, X, mixed_cor, eta)
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
            d2lp_deta2_zero_eta = ProfileLikelihood.likelihood_der2_eta(
                    z, X, mixed_cor, eta_zero)

            # method 2: using forward differencing from first derivative
            # dlp_deta_zero_eta = ProfileLikelihood.likelihood_der1_eta(
            #         z, X, mixed_cor, numpy.log10(eta_zero))
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
            sigma, sigma0 = ProfileLikelihood.find_optimal_sigma_sigma0(
                        z, X, mixed_cor, eta)

        # Output dictionary
        results = {
            'sigma': sigma,
            'sigma0': sigma0,
            'eta': eta,
            'iter': iter
        }

        return results

    # ===================
    # maximize likelihood
    # ===================

    @staticmethod
    def maximize_likelihood(
            z,
            X,
            cov,
            tol=1e-3,
            hyperparam_guess=[0.1, 0.1],
            optimization_method='Nelder-Mead',
            verbose=False):
        """
        Maximizing the log-likelihood function over the space of parameters
        sigma and sigma0

        In this function, hyperparam = [sigma, sigma0].
        """

        if optimization_method == 'chandrupatla':

            if len(hyperparam_guess) > 1:
                warnings.warn('"chandrupatla" method does not optimize ' +
                              '"distance_scale". The "distance scale in the ' +
                              'given "hyperparam_guess" will be ignored. To ' +
                              'optimize distance scale with "chandrupatla"' +
                              'method, set "profile_eta" to True.')
                distance_scale_guess = hyperparam_guess[1:]
                if cov.get_distance_scale() is None:
                    cov.set_distance_scale(distance_scale_guess)
                    warnings.warn('distance_scale is set based on the guess ' +
                                  'value.')

            # Note: When using interpolation, make sure the interval below is
            # exactly the end points of eta_i, not less or more.
            log_eta_guess = hyperparam_guess[0]
            min_eta_guess = numpy.min([1e-4, 10.0**log_eta_guess * 1e-2])
            max_eta_guess = numpy.max([1e+3, 10.0**log_eta_guess * 1e+2])
            interval_eta = [min_eta_guess, max_eta_guess]

            # Using root finding method on the first derivative w.r.t eta
            result = ProfileLikelihood.find_likelihood_der1_zeros(
                    z, X, cov.mixed_cor, interval_eta)

            # Finding the maxima. This isn't necessary and affects run time
            result['max_lp'] = ProfileLikelihood.likelihood(
                    z, X, cov.mixed_cor, False, result['eta'])

        else:
            # Partial function of likelihood (with minus to make maximization
            # to a minimization).
            sign_switch = True
            likelihood_partial_func = partial(
                    ProfileLikelihood.likelihood, z, X, cov.mixed_cor,
                    sign_switch)

            # Partial function of Jacobian of likelihood (with minus sign)
            jacobian_partial_func = partial(
                    ProfileLikelihood.likelihood_jacobian, z, X, cov,
                    sign_switch)

            # Partial function of Hessian of likelihood (with minus sign)
            # hessian_partial_func = partial(
            #         ProfileLikelihood.likelihood_hessian, z, X,
            #         cov, sign_switch)

            # Minimize
            res = scipy.optimize.minimize(likelihood_partial_func,
                                          hyperparam_guess,
                                          method=optimization_method, tol=tol,
                                          jac=jacobian_partial_func)
                                          # hess=hessian_partial_func)

            # print('Iter: %d, Eval: %d, Success: %s'
            #       % (res.nit, res.nfev, res.success))
            # print(res)

            # Extract res
            log_eta = res.x[0]
            if numpy.isneginf(log_eta):
                eta = 0.0
            else:
                eta = 10.0**log_eta
            # eta = log_eta   # Test
            sigma, sigma0 = ProfileLikelihood.find_optimal_sigma_sigma0(
                    z, X, cov.mixed_cor, eta)
            max_lp = -res.fun

            # Distance scale
            if res.x.size > 1:
                distance_scale = res.x[1:]
            else:
                distance_scale = cov.get_distance_scale()

            # Output dictionary
            result = {
                'sigma': sigma,
                'sigma0': sigma0,
                'eta': eta,
                'distance_scale': distance_scale,
                'max_lp': max_lp
            }

        return result

    # =============================
    # plot likelihood for fixed eta
    # =============================

    @staticmethod
    def plot_likelihood_for_fixed_eta(
            z,
            X,
            cov,
            eta_optimal,
            other_etas=None):
        """
        Plots log likelihood versus sigma, eta hyperparam
        """

        load_plot_settings()

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
            etas = numpy.r_[eta_optimal, other_etas]
        else:
            etas = numpy.r_[eta_optimal]
        etas = numpy.sort(etas)

        # Generate lp for various distance scales
        # distance_scale = numpy.logspace(-3, 3, 200)
        distance_scale = numpy.logspace(-3, 1, 500)
        d0_lp = numpy.zeros((etas.size, distance_scale.size), dtype=float)
        d1_lp = numpy.zeros((etas.size, distance_scale.size), dtype=float)
        d2_lp = numpy.zeros((etas.size, distance_scale.size), dtype=float)
        d1_lp_numerical = numpy.zeros((etas.size, distance_scale.size-2),
                                      dtype=float)
        d2_lp_numerical = numpy.zeros((etas.size, distance_scale.size-4),
                                      dtype=float)

        fig, ax = plt.subplots(ncols=3, figsize=(17, 5))
        colors = matplotlib.cm.nipy_spectral(
                numpy.linspace(0, 0.9, etas.size))

        for i in range(etas.size):
            for j in range(distance_scale.size):

                # Set the distance_scale
                cov.set_distance_scale(distance_scale[j])

                # Likelihood
                d0_lp[i, j] = ProfileLikelihood.likelihood(
                        z, X, cov.mixed_cor, False, numpy.log10(etas[i]))

                # First derivative of likelihood w.r.t distance scale
                hyperparam = numpy.r_[numpy.log10(etas[i]), distance_scale[j]]
                d1_lp[i, j] = ProfileLikelihood.likelihood_der1_distance_scale(
                        z, X, cov, hyperparam)

                # Second derivative of likelihood w.r.t distance scale
                d2_lp[i, j] = ProfileLikelihood.likelihood_der2_distance_scale(
                        z, X, cov, hyperparam)

            # Compute first derivative numerically
            d1_lp_numerical[i, :] = (d0_lp[i, 2:] - d0_lp[i, :-2]) / \
                (distance_scale[2:] - distance_scale[:-2])

            # Compute second derivative numerically
            d2_lp_numerical[i, :] = \
                (d1_lp_numerical[i, 2:] - d1_lp_numerical[i, :-2]) / \
                (distance_scale[3:-1] - distance_scale[1:-3])

            # Find maximum of lp
            max_index = numpy.argmax(d0_lp[i, :])
            optimal_distance_scale = distance_scale[max_index]
            optimal_lp = d0_lp[i, max_index]

            # Plot
            if etas[i] == eta_optimal:
                label = r'$\hat{\eta}=%0.2e$' % etas[i]
                marker = 'X'
            else:
                label = r'$\eta=%0.2e$' % etas[i]
                marker = 'o'
            ax[0].plot(distance_scale, d0_lp[i, :], color=colors[i],
                       label=label)
            ax[1].plot(distance_scale, d1_lp[i, :], color=colors[i],
                       label=label)
            ax[2].plot(distance_scale, d2_lp[i, :], color=colors[i],
                       label=label)
            ax[1].plot(distance_scale[1:-1], d1_lp_numerical[i, :], '--',
                       color=colors[i])
            ax[2].plot(distance_scale[2:-2], d2_lp_numerical[i, :], '--',
                       color=colors[i])
            p = ax[0].plot(optimal_distance_scale, optimal_lp, marker,
                           color=colors[i], markersize=3)
            ax[1].plot(optimal_distance_scale, 0.0,  marker,
                       color=colors[i], markersize=3)

        ax[0].legend(p, [r'optimal $\theta$'])
        ax[0].legend(loc='lower right')
        ax[1].legend(loc='lower right')
        ax[2].legend(loc='lower right')
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[2].set_xscale('log')

        # Plot annotations
        ax[0].set_xlim([distance_scale[0], distance_scale[-1]])
        ax[1].set_xlim([distance_scale[0], distance_scale[-1]])
        ax[2].set_xlim([distance_scale[0], distance_scale[-1]])
        ax[0].set_xlabel(r'$\theta$')
        ax[1].set_xlabel(r'$\theta$')
        ax[2].set_xlabel(r'$\theta$')
        ax[0].set_ylabel(r'$\ell(\theta | \eta)$')
        ax[1].set_ylabel(
            r'$\frac{\mathrm{d} \ell(\theta | \eta)}{\mathrm{d} \theta}$')
        ax[2].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} \theta^2}$')
        ax[0].set_title(r'Log likelihood function for fixed $\eta$')
        ax[1].set_title(r'First derivative of log likelihood function for ' +
                        r'fixed $\eta$')
        ax[2].set_title(r'Second derivative of log likelihood function for ' +
                        r'fixed $\eta$')
        ax[0].grid(True)
        ax[1].grid(True)
        ax[2].grid(True)

        plt.tight_layout()
        plt.show()

    # ========================================
    # plot likelihood for fixed distance scale
    # ========================================

    @staticmethod
    def plot_likelihood_for_fixed_distance_scale(
            z,
            X,
            mixed_cor,
            optimal_distance_scale,
            other_distance_scales=None):
        """
        Plots log likelihood versus sigma, eta hyperparam
        """

        load_plot_settings()

        # Convert distance_scale to a numpy array
        if other_distance_scales is not None:
            if numpy.isscalar(other_distance_scales):
                other_distance_scales = numpy.array([other_distance_scales])
            elif isinstance(other_distance_scales, list):
                other_distance_scales = numpy.array(other_distance_scales)
            elif not isinstance(other_distance_scales, numpy.ndarray):
                raise TypeError('"other_distance_scales" should be either a ' +
                                'scalar, list, or numpy.ndarray.')

        # Concatenate all given eta
        if other_distance_scales is not None:
            distance_scales = numpy.r_[optimal_distance_scale,
                                       other_distance_scales]
        else:
            distance_scales = numpy.r_[optimal_distance_scale]
        distance_scales = numpy.sort(distance_scales)

        eta = numpy.logspace(-3, 3, 200)
        d0_lp = numpy.zeros((distance_scales.size, eta.size,), dtype=float)
        d1_lp = numpy.zeros((distance_scales.size, eta.size,), dtype=float)
        d2_lp = numpy.zeros((distance_scales.size, eta.size,), dtype=float)
        d1_lp_numerical = numpy.zeros((distance_scales.size, eta.size-2,),
                                      dtype=float)
        d2_lp_numerical = numpy.zeros((distance_scales.size, eta.size-4,),
                                      dtype=float)

        fig, ax = plt.subplots(ncols=3, figsize=(17, 5))
        colors = matplotlib.cm.nipy_spectral(
                numpy.linspace(0, 0.9, distance_scales.size))

        for i in range(distance_scales.size):
            mixed_cor.set_distance_scale(distance_scales[i])
            for j in range(eta.size):

                # Likelihood function
                d0_lp[i, j] = ProfileLikelihood.likelihood(
                        z, X, mixed_cor, False, numpy.log10(eta[j]))

                # First derivative w.r.t eta
                d1_lp[i, j] = ProfileLikelihood.likelihood_der1_eta(
                        z, X, mixed_cor, numpy.log10(eta[j]))

                # Second derivative w.r.t eta
                d2_lp[i, j] = ProfileLikelihood.likelihood_der2_eta(
                        z, X, mixed_cor, numpy.log10(eta[j]))

            # Compute first derivative numerically
            d1_lp_numerical[i, :] = (d0_lp[i, 2:] - d0_lp[i, :-2]) / \
                (eta[2:] - eta[:-2])

            # Compute second derivative numerically
            d2_lp_numerical[i, :] = \
                (d1_lp_numerical[i, 2:] - d1_lp_numerical[i, :-2]) / \
                (eta[3:-1] - eta[1:-3])

            # Find maximum of lp
            max_index = numpy.argmax(d0_lp[i, :])
            optimal_eta = eta[max_index]
            optimal_lp = d0_lp[i, max_index]

            if distance_scales[i] == optimal_distance_scale:
                label = r'$\hat{\theta} = %0.2e$' % distance_scales[i]
                marker = 'X'
            else:
                label = r'$\theta = %0.2e$' % distance_scales[i]
                marker = 'o'

            ax[0].plot(eta, d0_lp[i, :], color=colors[i], label=label)
            ax[1].plot(eta, d1_lp[i, :], color=colors[i], label=label)
            ax[2].plot(eta, d2_lp[i, :], color=colors[i], label=label)
            ax[1].plot(eta[1:-1], d1_lp_numerical[i, :], '--', color=colors[i])
            ax[2].plot(eta[2:-2], d2_lp_numerical[i, :], '--', color=colors[i])

            p = ax[0].plot(optimal_eta, optimal_lp, marker, color=colors[i],
                           markersize=3)
            ax[1].plot(optimal_eta, 0.0, marker, color=colors[i], markersize=3)

        ax[0].legend(p, [r'optimal $\eta$'])
        ax[0].legend(loc='lower right')
        ax[1].legend(loc='lower right')
        ax[2].legend(loc='lower right')

        # Plot annotations
        ax[0].set_xlim([eta[0], eta[-1]])
        ax[1].set_xlim([eta[0], eta[-1]])
        ax[2].set_xlim([eta[0], eta[-1]])
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[2].set_xscale('log')
        ax[0].set_xlabel(r'$\eta$')
        ax[1].set_xlabel(r'$\eta$')
        ax[2].set_xlabel(r'$\eta$')
        ax[0].set_ylabel(r'$\ell(\eta | \theta)$')
        ax[1].set_ylabel(
            r'$\frac{\mathrm{d}\ell(\eta | \theta)}{\mathrm{d}\eta}$')
        ax[2].set_ylabel(
            r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d}\eta^2}$')
        ax[0].set_title(r'Log likelihood function for fixed $\theta$')
        ax[1].set_title(r'First derivative of log likelihood function for ' +
                        r'fixed $\theta$')
        ax[2].set_title(r'Second derivative of log likelihood function for ' +
                        r'fixed $\theta$')
        ax[0].grid(True)
        ax[1].grid(True)
        ax[2].grid(True)

        plt.tight_layout()
        plt.show()

    # ===============
    # plot likelihood
    # ===============

    @staticmethod
    def plot_likelihood(z, X, mixed_cor, result):
        """
        Plots log likelihood versus sigma, eta hyperparam
        """

        load_plot_settings()

        #  Optimal point
        optimal_eta = result['eta']
        optimal_distance_scale = result['distance_scale']
        optimal_lp = result['max_lp']

        eta = numpy.logspace(-3, 3, 100)
        distance_scale = numpy.logspace(-3, 3, 100)
        lp = numpy.zeros((distance_scale.size, eta.size), dtype=float)

        # Compute lp
        for i in range(distance_scale.size):
            mixed_cor.set_distance_scale(distance_scale[i])
            for j in range(eta.size):
                lp[i, j] = ProfileLikelihood.likelihood(
                        z, X, mixed_cor, False, numpy.log10(eta[j]))

        # Convert inf to nan
        lp = numpy.where(numpy.isinf(lp), numpy.nan, lp)

        [distance_scale_mesh, eta_mesh] = numpy.meshgrid(distance_scale, eta)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(numpy.log10(eta_mesh),
                               numpy.log10(distance_scale_mesh), lp.T,
                               linewidth=0, antialiased=True, alpha=0.9,
                               label=r'$\ell(\eta, \theta)$')
        fig.colorbar(surf, ax=ax)

        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # Find max for each fixed eta
        opt_distance_scale1 = numpy.zeros((eta.size, ), dtype=float)
        opt_lp1 = numpy.zeros((eta.size, ), dtype=float)
        opt_lp1[:] = numpy.nan
        for j in range(eta.size):
            if numpy.all(numpy.isnan(lp[:, j])):
                continue
            max_index = numpy.nanargmax(lp[:, j])
            opt_distance_scale1[j] = distance_scale[max_index]
            opt_lp1[j] = lp[max_index, j]
        ax.plot3D(numpy.log10(eta), numpy.log10(opt_distance_scale1), opt_lp1,
                  color='red', label=r'$\max_{\theta} \ell(\theta | \eta)$')

        # Find max for each fixed distance_scale
        opt_eta2 = numpy.zeros((distance_scale.size, ), dtype=float)
        opt_lp2 = numpy.zeros((distance_scale.size, ), dtype=float)
        opt_lp2[:] = numpy.nan
        for i in range(distance_scale.size):
            if numpy.all(numpy.isnan(lp[i, :])):
                continue
            max_index = numpy.nanargmax(lp[i, :])
            opt_eta2[i] = eta[max_index]
            opt_lp2[i] = lp[i, max_index]
        ax.plot3D(numpy.log10(opt_eta2), numpy.log10(distance_scale), opt_lp2,
                  color='goldenrod',
                  label=r'$\max_{\eta} \ell(\eta | \theta)$')

        # Plot max of the whole 2D array
        max_indices = numpy.unravel_index(numpy.nanargmax(lp), lp.shape)
        opt_distance_scale = distance_scale[max_indices[0]]
        opt_eta = eta[max_indices[1]]
        opt_lp = lp[max_indices[0], max_indices[1]]
        ax.plot3D(numpy.log10(opt_eta), numpy.log10(opt_distance_scale),
                  opt_lp, 'o', color='red', markersize=6,
                  label=r'$\max_{\eta, \theta} \ell$ (by brute force on grid)')

        # Plot optimal point as found by the profile likelihood method
        ax.plot3D(numpy.log10(optimal_eta),
                  numpy.log10(optimal_distance_scale),
                  optimal_lp, 'o', color='magenta', markersize=6,
                  label=r'$\max_{\eta, \theta} \ell$ (by optimization)')

        # Plot annotations
        ax.legend()
        ax.set_xlim([numpy.log10(eta[0]), numpy.log10(eta[-1])])
        ax.set_ylim([numpy.log10(distance_scale[0]),
                    numpy.log10(distance_scale[-1])])
        ax.set_xlabel(r'$\log_{10}(\eta)$')
        ax.set_ylabel(r'$\log_{10}(\theta)$')
        ax.set_zlabel(r'$\ell(\eta, \theta)$')
        ax.set_title('Log Likelihood function')
        plt.show()

    # =======================
    # compute bounds der1 eta
    # =======================

    @staticmethod
    def compute_bounds_der1_eta(X, K, eta):
        """
        Upper and lower bound.
        """

        n, m = X.shape
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

    @staticmethod
    def compute_asymptote_der1_eta(z, X, K, eta):
        """
        Computes first and second order asymptote to the first derivative of
        log marginal likelihood function.
        """

        # Initialize output
        asymptote_1_order = numpy.empty(eta.size)
        asymptote_2_order = numpy.empty(eta.size)

        n, m = X.shape
        I = numpy.eye(n)                                           # noqa: E741
        # Im = numpy.eye(m)
        Q = X@numpy.linalg.inv(X.T@X)@X.T
        R = I - Q
        N = K@R
        N2 = N@N
        N3 = N2@N
        N4 = N3@N

        mtrN = numpy.trace(N)/(n-m)
        mtrN2 = numpy.trace(N2)/(n-m)

        A0 = -R@(mtrN*I - N)
        A1 = R@(mtrN*N + mtrN2*I - 2*N2)
        A2 = -R@(mtrN*N2 + mtrN2*N - 2*N3)
        A3 = R@(mtrN2*N2 - N4)

        zRz = numpy.dot(z, numpy.dot(R, z))
        z_Rnorm = numpy.sqrt(zRz)
        zc = z / z_Rnorm

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

    @staticmethod
    def plot_likelihood_der1_eta(z, X, mixed_cor, optimal_eta):
        """
        Plots the derivative of log likelihood as a function of eta.
        Also it shows where the optimal eta is, which is the location
        where the derivative is zero.
        """

        print('Plot first derivative ...')

        load_plot_settings()

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
            dlp_deta[i] = ProfileLikelihood.likelihood_der1_eta(
                    z, X, mixed_cor, numpy.log10(eta[i]))

        # Compute upper and lower bound of derivative
        K = mixed_cor.get_matrix(0.0)
        dlp_deta_upper_bound, dlp_deta_lower_bound = \
            ProfileLikelihood.compute_bounds_der1_eta(X, K, eta)

        # Compute asymptote of first derivative, using both first and second
        # order approximation
        try:
            # eta_high_res might not be defined, depending on plot_optimal_eta
            x = eta_high_res
        except NameError:
            x = numpy.logspace(1, log_eta_end, 100)
        dlp_deta_asymptote_1, dlp_deta_asymptote_2, roots_1, roots_2 = \
            ProfileLikelihood.compute_asymptote_der1_eta(z, X, K, x)

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
