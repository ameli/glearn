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
import scipy.optimize
from functools import partial
from .._utilities.plot_utilities import *                    # noqa: F401, F403
from .._utilities.plot_utilities import load_plot_settings, save_plot, plt
from ._likelihood_utilities import M_dot
import imate


# ===============
# Full Likelihood
# ===============

class FullLikelihood(object):

    # ==========
    # likelihood
    # ==========

    @staticmethod
    def likelihood(z, X, cov, sign_switch, hyperparam):
        """
        Here we use direct parameter, sigma and sigma0

        sign_switch change s the sign of the output from lp to -lp. When True,
        this is used to minimizing (instead of maximizing) the negative of
        log-likelihood function.
        """

        # hyperparameters
        sigma = hyperparam[0]
        sigma0 = hyperparam[1]

        # Include derivative w.r.t distance_scale
        if hyperparam.size > 2:
            distance_scale = numpy.abs(hyperparam[2:])
            cov.set_distance_scale(distance_scale)

        n, m = X.shape

        # cov is the (sigma**2) * K + (sigma0**2) * I
        logdet_S = cov.logdet(sigma, sigma0)
        Y = cov.solve(sigma, sigma0, X)

        # Compute zMz
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)
        Mz = M_dot(cov, Binv, Y, sigma, sigma0, z)
        zMz = numpy.dot(z, Mz)

        # Compute log det (X.T*Sinv*X)
        logdet_B = numpy.log(numpy.linalg.det(B))

        # Log likelihood
        lp = -0.5*(n-m)*numpy.log(2.0*numpy.pi) - 0.5*logdet_S \
             - 0.5*logdet_B - 0.5*zMz

        # If lp is used in scipy.optimize.minimize, change the sign to obtain
        # the minimum of -lp
        if sign_switch:
            lp = -lp

        return lp

    # ===================
    # likelihood jacobian
    # ===================

    @staticmethod
    def likelihood_jacobian(z, X, cov, sign_switch, hyperparam):
        """
        When both :math:`\\sigma` and :math:`\\sigma_0` are zero, jacobian is
        undefined.
        """

        # hyperparameters
        sigma = hyperparam[0]
        sigma0 = hyperparam[1]

        # Include derivative w.r.t distance_scale
        if hyperparam.size > 2:
            distance_scale = numpy.abs(hyperparam[2:])
            cov.set_distance_scale(distance_scale)

        n, m = X.shape

        # Computing Y=Sinv*X and w=Sinv*z.
        Y = cov.solve(sigma, sigma0, X)

        # B is Xt * Y
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)

        # Compute Mz
        Mz = M_dot(cov, Binv, Y, sigma, sigma0, z)

        # Compute KMz (Setting sigma=1 and sigma0=0 to have cov = K)
        KMz = cov.dot(1.0, 0.0, Mz)

        # Compute zMMz and zMKMz
        zMMz = numpy.dot(Mz, Mz)
        zMKMz = numpy.dot(Mz, KMz)

        # Compute trace of M
        if numpy.abs(sigma) < cov.tol:
            trace_M = (n - m) / sigma0**2
        else:
            trace_Sinv = cov.traceinv(sigma, sigma0)
            YtY = numpy.matmul(Y.T, Y)
            trace_BinvYtY = numpy.trace(numpy.matmul(Binv, YtY))
            trace_M = trace_Sinv - trace_BinvYtY

        # Compute trace of KM which is (n-m)/sigma**2 - eta* trace(M)
        if numpy.abs(sigma) < cov.tol:
            YtKY = numpy.matmul(Y.T, cov.dot(1.0, 0.0, Y))
            BinvYtKY = numpy.matmul(Binv, YtKY)
            trace_BinvYtKY = numpy.trace(BinvYtKY)
            trace_KM = n/sigma0**2 - trace_BinvYtKY
        else:
            eta = (sigma0 / sigma)**2
            trace_KM = (n - m)/sigma**2 - eta*trace_M

        # Derivative of lp wrt to sigma
        der1_sigma = -0.5*trace_KM + 0.5*zMKMz
        der1_sigma0 = -0.5*trace_M + 0.5*zMMz

        jacobian = numpy.array([der1_sigma, der1_sigma0], dtype=float)

        # Compute Jacobian w.r.t distance_scale
        if hyperparam.size > 2:

            der1_distance_scale = numpy.zeros((distance_scale.size, ),
                                              dtype=float)

            # Needed to compute trace (TODO)
            S = cov.get_matrix(sigma, sigma0)
            Sinv = numpy.linalg.inv(S)

            # Sp is the derivative of cov w.r.t the p-th element of
            # distance_scale.
            for p in range(distance_scale.size):

                # Compute zMSpMz
                SpMz = cov.dot(sigma, sigma0, Mz, derivative=[p])
                zMSpMz = numpy.dot(Mz, SpMz)

                # Compute the first component of trace of Sp * Sinv (TODO)
                Sp = cov.get_matrix(sigma, sigma0, derivative=[p])
                SpSinv = Sp @ Sinv
                trace_SpSinv, _ = imate.trace(SpSinv, method='exact')

                # Compute the second component of trace of Sp * M
                SpY = cov.dot(sigma, sigma0, Y, derivative=[p])
                YtSpY = numpy.matmul(Y.T, SpY)
                BinvYtSpY = numpy.matmul(Binv, YtSpY)
                trace_BinvYtSpY = numpy.trace(BinvYtSpY)

                # Compute trace of Sp * M
                trace_SpM = trace_SpSinv - trace_BinvYtSpY

                # Derivative of lp w.r.t p-th element of distance scale
                der1_distance_scale[p] = -0.5*trace_SpM + 0.5*zMSpMz

            # Concatenate jacobian
            jacobian = numpy.r_[jacobian, der1_distance_scale]

        # Test
        # print(jacobian)

        if sign_switch:
            jacobian = -jacobian

        return jacobian

    # ==================
    # likelihood hessian
    # ==================

    @staticmethod
    def likelihood_hessian(z, X, cov, sign_switch, hyperparam):
        """
        """

        # hyperparameters
        sigma = hyperparam[0]
        sigma0 = hyperparam[1]
        eta = (sigma0 / sigma)**2

        # Include derivatove w.r.t distance_scale
        if hyperparam.size > 2:
            distance_scale = numpy.abs(hyperparam[2:])
            cov.set_distance_scale(distance_scale)

        n, m = X.shape

        # -----------------------------------------
        # Second derivatives w.r.t sigma and sigma0
        # -----------------------------------------

        # Computing Y=Sinv*X, V = Sinv*Y, and w=Sinv*z
        Y = cov.solve(sigma, sigma0, X)
        V = cov.solve(sigma, sigma0, Y)

        # B is Xt * Y
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)
        YtY = numpy.matmul(Y.T, Y)
        A = numpy.matmul(Binv, YtY)

        # Compute Mz, MMz
        Mz = M_dot(cov, Binv, Y, sigma, sigma0, z)
        MMz = M_dot(cov, Binv, Y, sigma, sigma0, Mz)

        # Compute KMz, zMMMz (Setting sigma=1 and sigma0=0 to have cov=K)
        KMz = cov.dot(1.0, 0.0, Mz)
        zMMMz = numpy.dot(Mz, MMz)

        # Compute MKMz
        MKMz = M_dot(cov, Binv, Y, sigma, sigma0, KMz)

        # Compute zMKMKMz
        zMMKMz = numpy.dot(MMz, KMz)
        zMKMKMz = numpy.dot(KMz, MKMz)

        # Trace of M
        if numpy.abs(sigma) < cov.tol:
            trace_M = (n - m) / sigma0**2
        else:
            trace_Sinv = cov.traceinv(sigma, sigma0)
            trace_A = numpy.trace(A)
            trace_M = trace_Sinv - trace_A

        # Trace of Sinv**2
        trace_S2inv = cov.traceinv(sigma, sigma0, exponent=2)

        # Trace of M**2
        YtV = numpy.matmul(Y.T, V)
        C = numpy.matmul(Binv, YtV)
        trace_C = numpy.trace(C)
        AA = numpy.matmul(A, A)
        trace_AA = numpy.trace(AA)
        trace_M2 = trace_S2inv - 2.0*trace_C + trace_AA

        # Trace of (KM)**2
        if numpy.abs(sigma) < cov.tol:
            trace_K2 = cov.trace(1.0, 0.0, exponent=2)
            D = numpy.matmul(X.T, X)
            Dinv = numpy.linalg.inv(D)
            KX = cov.dot(1.0, 0.0, X, exponent=1)
            XKX = numpy.matmul(X.T, KX)
            XK2X = numpy.matmul(KX.T, KX)
            E = numpy.matmul(Dinv, XKX)
            E2 = numpy.matmul(E, E)
            F = numpy.matmul(Dinv, XK2X)
            trace_KMKM = (trace_K2 - 2.0*numpy.trace(F) + numpy.trace(E2)) / \
                sigma0**4
        else:
            trace_KMKM = (n-m)/sigma**4 - (2*eta/sigma**2)*trace_M + \
                (eta**2)*trace_M2

        # Trace of K*(M**2)
        if numpy.abs(sigma) < cov.tol:
            trace_KM = (n - numpy.trace(E))/sigma0**2
            trace_KMM = trace_KM / sigma0**2
        else:
            trace_KMM = trace_M/sigma**2 - eta*trace_M2

        # Compute second derivatives
        der2_sigma0_sigma0 = 0.5*trace_M2 - zMMMz
        der2_sigma_sigma = 0.5*trace_KMKM - zMKMKMz
        der2_sigma_sigma0 = 0.5*trace_KMM - zMMKMz

        # Hessian
        hessian = numpy.array(
                [[der2_sigma_sigma, der2_sigma_sigma0],
                 [der2_sigma_sigma0, der2_sigma0_sigma0]], dtype=float)

        # Compute Hessian w.r.t distance_scale
        if hyperparam.size > 2:

            # Initialize arrays
            der2_distance_scale = numpy.zeros(
                    (distance_scale.size, distance_scale.size), dtype=float)
            der2_sigma_distance_scale = numpy.zeros((distance_scale.size),
                                                    dtype=float)
            der2_sigma0_distance_scale = numpy.zeros((distance_scale.size),
                                                     dtype=float)

            # Needed to compute trace (TODO)
            S = cov.get_matrix(sigma, sigma0)
            Sinv = numpy.linalg.inv(S)

            # Sp is the derivative of cov w.r.t the p-th element of
            # distance_scale. Spq is the second mixed derivative of S w.r.t
            # p-th and q-th elements of distance_scale.
            for p in range(distance_scale.size):

                # --------------------------------------------------------
                # 1. Compute mixed derivatives of distance_scale and sigma
                # --------------------------------------------------------

                # 1.1. Compute zMSpMKMz
                SpMz = cov.dot(sigma, sigma0, Mz, derivative=[p])
                MSpMz = M_dot(cov, Binv, Y, sigma, sigma0, SpMz)
                zMSpMKMz = numpy.dot(MSpMz, MKMz)

                # 1.2. Compute zMKpMz
                KpMz = cov.dot(1.0, 0.0, Mz, derivative=[p])
                zMKpMz = numpy.dot(Mz, KpMz)

                # 1.3. Compute trace of Kp * M

                # Compute the first component of trace of Kp * Sinv (TODO)
                Kp = cov.get_matrix(1.0, 0.0, derivative=[p])
                KpSinv = numpy.matmul(Kp, Sinv)
                trace_KpSinv, _ = imate.trace(KpSinv, method='exact')

                # Compute the second component of trace of Kp * M
                KpY = cov.dot(1.0, 0.0, Y, derivative=[p])
                YtKpY = numpy.matmul(Y.T, KpY)
                BinvYtKpY = numpy.matmul(Binv, YtKpY)
                trace_BinvYtKpY = numpy.trace(BinvYtKpY)

                # Compute trace of Kp * M
                trace_KpM = trace_KpSinv - trace_BinvYtKpY

                # 1.4. Compute trace of K * M * Sp * M

                # Compute first part of trace of K * M * Sp * M
                K = cov.get_matrix(1.0, 0.0, derivative=[])
                KSinv = numpy.matmul(K, Sinv)
                Sp = cov.get_matrix(sigma, sigma0, derivative=[p])
                SpSinv = numpy.matmul(Sp, Sinv)
                KSinvSpSinv = numpy.matmul(KSinv, SpSinv)
                trace_KMSpM_1, _ = imate.trace(KSinvSpSinv, method='exact')

                # Compute the second part of trace of K * M * Sp * M
                KY = numpy.matmul(K, Y)
                SpY = numpy.matmul(Sp, Y)
                SinvSpY = cov.solve(sigma, sigma0, SpY)
                YtKSinvSpY = numpy.matmul(KY.T, SinvSpY)
                C21 = numpy.matmul(Binv, YtKSinvSpY)
                C22 = numpy.matmul(Binv, YtKSinvSpY.T)
                trace_KMSpM_21 = numpy.trace(C21)
                trace_KMSpM_22 = numpy.trace(C22)

                # Compute the third part of trace of K * M * Sp * M
                YtKY = numpy.matmul(Y.T, KY)
                YtSpY = numpy.matmul(Y.T, SpY)
                Dk = numpy.matmul(Binv, YtKY)
                Dp = numpy.matmul(Binv, YtSpY)
                D = numpy.matmul(Dk, Dp)
                trace_KMSpM_3 = numpy.trace(D)

                # Compute trace of K * M * Sp * M
                trace_KMSpM = trace_KMSpM_1 - trace_KMSpM_21 - \
                    trace_KMSpM_22 + trace_KMSpM_3

                # 1.5. Second derivatives w.r.t distance_scale
                der2_sigma_distance_scale[p] = -0.5*trace_KpM + \
                    0.5*trace_KMSpM - zMSpMKMz + 0.5*zMKpMz

                # ---------------------------------------------------------
                # 2. Compute mixed derivatives of distance_scale and sigma0
                # ---------------------------------------------------------

                # 2.1. Compute zMSpMMz
                zMSpMMz = numpy.dot(MSpMz, MMz)

                # 2.4. Compute trace of M * Sp * M

                # Compute first part of trace of M * Sp * M
                SinvSpSinv = numpy.matmul(Sinv, SpSinv)
                trace_MSpM_1, _ = imate.trace(SinvSpSinv, method='exact')

                # Compute the second part of trace of M * Sp * M
                YtSinvSpY = numpy.matmul(Y.T, SinvSpY)
                C21 = numpy.matmul(Binv, YtSinvSpY)
                C22 = numpy.matmul(Binv, YtSinvSpY.T)
                trace_MSpM_21 = numpy.trace(C21)
                trace_MSpM_22 = numpy.trace(C22)

                # Compute the third part of trace of M * Sp * M
                D = numpy.matmul(Dp, A)
                trace_MSpM_3 = numpy.trace(D)

                # Compute trace of M * Sp * M
                trace_MSpM = trace_MSpM_1 - trace_MSpM_21 - trace_MSpM_22 + \
                    trace_MSpM_3

                # 2.5. Second derivatives w.r.t distance_scale
                der2_sigma0_distance_scale[p] = 0.5*trace_MSpM - zMSpMMz

                # Concatenate mixed derivatives of distance_scale and sigmas
                der2_mixed = numpy.c_[der2_sigma_distance_scale,
                                      der2_sigma0_distance_scale]

                # --------------------------------------------
                # Compute second derivatives of distance_scale
                # --------------------------------------------

                for q in range(p, distance_scale.size):

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
                    SpqSinv = numpy.matmul(Spq, Sinv)
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
                    SpSinv = numpy.matmul(Sp, Sinv)
                    Sq = cov.get_matrix(sigma, sigma0, derivative=[q])
                    if p == q:
                        SqSinv = SpSinv
                    else:
                        SqSinv = numpy.matmul(Sq, Sinv)
                    SpSinvSqSinv = numpy.matmul(SpSinv, SqSinv)
                    trace_SpMSqM_1, _ = imate.trace(SpSinvSqSinv,
                                                    method='exact')

                    # Compute the second part of trace of Sp * M * Sq * M
                    SpY = numpy.matmul(Sp, Y)
                    if p == q:
                        SqY = SpY
                    else:
                        SqY = numpy.matmul(Sq, Y)
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

            # ---------------------------------
            # Concatenate all mixed derivatives
            # ---------------------------------

            hessian = numpy.block([[hessian, der2_mixed],
                                   [der2_mixed.T, der2_distance_scale]])

        if sign_switch:
            hessian = -hessian

        return hessian

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
            optimization_method='Nelder-Mead'):
        """
        Maximizing the log-likelihood function over the space of parameters
        sigma and sigma0

        In this function, hyperparam = [sigma, sigma0].
        """

        # Keeping times
        initial_wall_time = time.time()
        initial_proc_time = time.process_time()

        # Partial function of likelihood (with minus to make maximization to a
        # minimization).
        sign_switch = True
        likelihood_partial_func = partial(
                FullLikelihood.likelihood, z, X, cov, sign_switch)

        # Partial function of Jacobian of likelihood (with minus sign)
        jacobian_partial_func = partial(
                FullLikelihood.likelihood_jacobian, z, X, cov,
                sign_switch)

        # Partial function of Hessian of likelihood (with minus sign)
        hessian_partial_func = partial(
                FullLikelihood.likelihood_hessian, z, X, cov,
                sign_switch)

        # Minimize
        res = scipy.optimize.minimize(likelihood_partial_func,
                                      hyperparam_guess,
                                      method=optimization_method, tol=tol,
                                      jac=jacobian_partial_func,
                                      hess=hessian_partial_func)

        print(res)

        print('Iter: %d, Eval: %d, Success: %s'
              % (res.nit, res.nfev, res.success))

        # Extract res
        sigma = res.x[0]
        sigma0 = res.x[1]
        eta = (sigma0/sigma)**2
        max_lp = -res.fun

        # Distance scale
        if res.x.size > 1:
            distance_scale = numpy.abs(res.x[1:])
        else:
            distance_scale = cov.get_distance_scale()

        # Adding time to the results
        wall_time = time.time() - initial_wall_time
        proc_time = time.process_time() - initial_proc_time

        # Output dictionary
        result = {
            'hyperparam':
            {
                'sigma': sigma,
                'sigma0': sigma0,
                'eta': eta,
                'distance_scale': distance_scale,
            },
            'optimization':
            {
                'max_likelihood': max_lp,
                'iter': res.nit,
            },
            'time':
            {
                'wall_time': wall_time,
                'proc_time': proc_time
            }
        }

        return result

    # ===============
    # plot likelihood
    # ===============

    @staticmethod
    def plot_likelihood(z, X, cov, result=None):
        """
        Plots log likelihood versus sigma0, sigma hyperparam
        """

        print('Plotting log likelihood ...')
        load_plot_settings()

        sigma0 = numpy.linspace(0.15, 0.25, 20)
        sigma = numpy.linspace(0, 0.1, 20)
        lp = numpy.zeros((sigma0.size, sigma.size))
        for i in range(sigma0.size):
            for j in range(sigma.size):
                lp[i, j] = FullLikelihood.likelihood(
                        z, X, cov, False, [sigma[j], sigma0[i]])

        [sigma_mesh, sigma0_mesh] = numpy.meshgrid(sigma, sigma0)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        p = ax.plot_surface(sigma_mesh, sigma0_mesh, lp, linewidth=0,
                            antialiased=False)
        fig.colorbar(p, ax=ax)

        if result is not None:
            opt_sigma = result['sigma']
            opt_sigma0 = result['sigma0']
            hyperparam = [opt_sigma, opt_sigma0]
            opt_lp = FullLikelihood.likelihood(
                        z, X, cov, False, hyperparam)
            plt.plot(opt_sigma, opt_sigma0, opt_lp, markersize=5, marker='o',
                     markerfacecolor='red', markeredgecolor='red')
        ax.set_xlabel(r'$\sigma$')
        ax.set_ylabel(r'$\sigma_0$')
        ax.set_title('Log Likelihood function')

        filename = 'likelihood'
        save_plot(plt, filename, transparent_background=False, pdf=True)

        plt.show()