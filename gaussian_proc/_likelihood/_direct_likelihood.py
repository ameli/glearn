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
import scipy.optimize
from functools import partial
from .._utilities.plot_utilities import *                    # noqa: F401, F403
from .._utilities.plot_utilities import load_plot_settings, save_plot, plt


# =================
# Direct Likelihood
# =================

class DirectLikelihood(object):

    # ==============
    # log likelihood
    # ==============

    @staticmethod
    def log_likelihood(z, X, K_mixed, sign_switch, hyperparam):
        """
        Here we use direct parameter, sigma and sigma0

        sign_switch chnages the sign of the output from lp to -lp. When True,
        this is used to minimizing (instad of maximizing) the negative of
        log-likelihood function.
        """

        # hyperparameters
        sigma = hyperparam[0]
        sigma0 = hyperparam[1]

        n, m = X.shape

        # S is the (sigma**2) * K + (sigma0**2) * I, but we don't construct it.
        # Instead, we consruct Kn = K + eta I, where eta = (sigma0 / sigma)**2
        tol = 1e-8
        if numpy.abs(sigma) < tol:

            # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
            logdet_S = n * numpy.log(sigma0**2)

            Y = X / sigma0**2

        else:
            eta = (sigma0 / sigma)**2
            logdet_Kn = K_mixed.logdet(eta)
            logdet_S = n * numpy.log(sigma**2) + logdet_Kn

            Y = K_mixed.solve(eta, X) / sigma**2

        # Compute log det (X.T*Sinv*X)
        XtSinvX = numpy.matmul(X.T, Y)
        logdet_XtSinvX = numpy.log(numpy.linalg.det(XtSinvX))

        # Compute zMz
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)
        Mz = DirectLikelihood.M_dot(K_mixed, Binv, Y, sigma, sigma0, z)
        zMz = numpy.dot(z, Mz)

        # Log likelihood
        lp = -0.5*(n-m)*numpy.log(2.0*numpy.pi) - 0.5*logdet_S \
             - 0.5*logdet_XtSinvX - 0.5*zMz

        # If lp is used in scipy.optimize.minimize, change the sign to optain
        # the minimum of -lp
        if sign_switch:
            lp = -lp

        return lp

    # =======================
    # log likelihood jacobian
    # =======================

    @staticmethod
    def log_likelihood_jacobian(z, X, K_mixed, sign_switch, hyperparam):
        """
        When both :math:`\\sigma` and :math:`\\sigma_0` are zero, jacobian is
        undefined.
        """

        # hyperparameters
        sigma = hyperparam[0]
        sigma0 = hyperparam[1]

        n, m = X.shape

        # S is the (sigma**2) * K + (sigma0**2) * I, but we don't construct it
        # Instead, we construct Kn = K + eta I, where eta = (sigma0 / sigma)**2

        # Computing Y=Sinv*X and w=Sinv*z
        tol = 1e-8
        if numpy.abs(sigma) < tol:

            # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
            Y = X / sigma0**2

        else:
            eta = (sigma0 / sigma)**2
            Y = K_mixed.solve(eta, X) / sigma**2

        # B is Xt * Y
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)

        # Compute Mz
        Mz = DirectLikelihood.M_dot(K_mixed, Binv, Y, sigma, sigma0, z)

        # Compute KMz
        KMz = K_mixed.dot(0, Mz)

        # Compute zMMz and zMKMz
        zMMz = numpy.dot(Mz, Mz)
        zMKMz = numpy.dot(Mz, KMz)

        # Compute trace of M
        if numpy.abs(sigma) < tol:
            trace_M = (n - m) / sigma0**2
        else:
            trace_Sinv = K_mixed.traceinv(eta) / sigma**2
            YtY = numpy.matmul(Y.T, Y)
            trace_BinvYtY = numpy.trace(numpy.matmul(Binv, YtY))
            trace_M = trace_Sinv - trace_BinvYtY

        # Compute trace of KM which is (n-m)/sigma**2 - eta* trace(M)
        if numpy.abs(sigma) < tol:
            YtKY = numpy.matmul(Y.T, K_mixed.dot(0, Y))
            BinvYtKY = numpy.matmul(Binv, YtKY)
            trace_BinvYtKY = numpy.trace(BinvYtKY)
            trace_KM = K_mixed.trace(0)/sigma0**2 - trace_BinvYtKY
        else:
            trace_KM = (n - m)/sigma**2 - eta*trace_M

        # Derivative of lp wrt to sigma
        der1_sigma = -0.5*trace_KM + 0.5*zMKMz
        der1_sigma0 = -0.5*trace_M + 0.5*zMMz

        jacobian = numpy.array([der1_sigma, der1_sigma0], dtype=float)

        if sign_switch:
            jacobian = -jacobian

        return jacobian

    # ======================
    # log likelihood hessian
    # ======================

    @staticmethod
    def log_likelihood_hessian(z, X, K_mixed, sign_switch, hyperparam):
        """
        """

        # hyperparameters
        sigma = hyperparam[0]
        sigma0 = hyperparam[1]

        n, m = X.shape

        # S is the (sigma**2) * K + (sigma0**2) * I, but we don't construct it
        # Instead, we construct Kn = K + eta I, where eta = (sigma0 / sigma)**2

        # Computing Y=Sinv*X, V = Sinv*Y, and w=Sinv*z
        # tol = 1e-8
        tol = 1e-16
        if numpy.abs(sigma) < tol:

            # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
            Y = X / sigma0**2
            V = Y / sigma0**2

        else:
            eta = (sigma0 / sigma)**2
            Y = K_mixed.solve(eta, X) / sigma**2
            V = K_mixed.solve(eta, Y) / sigma**2

        # B is Xt * Y
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)
        YtY = numpy.matmul(Y.T, Y)
        A = numpy.matmul(Binv, YtY)

        # Compute Mz, MMz
        Mz = DirectLikelihood.M_dot(K_mixed, Binv, Y, sigma, sigma0, z)
        MMz = DirectLikelihood.M_dot(K_mixed, Binv, Y, sigma, sigma0, Mz)

        # Compute KMz, zMMMz
        KMz = K_mixed.dot(0, Mz)
        zMMMz = numpy.dot(Mz, MMz)

        # Compute MKMz
        MKMz = DirectLikelihood.M_dot(K_mixed, Binv, Y, sigma, sigma0, KMz)

        # Compute zMKMKMz
        zMMKMz = numpy.dot(MMz, KMz)
        zMKMKMz = numpy.dot(KMz, MKMz)

        # Trace of M
        if numpy.abs(sigma) < tol:
            trace_M = (n - m) / sigma0**2
        else:
            trace_Sinv = K_mixed.traceinv(eta) / sigma**2
            trace_A = numpy.trace(A)
            trace_M = trace_Sinv - trace_A

        # Trace of Sinv**2
        if numpy.abs(sigma) < tol:
            trace_S2inv = n / sigma0**4
        else:
            trace_S2inv = K_mixed.traceinv(eta, exponent=2) / sigma**4

        # Trace of M**2
        YtV = numpy.matmul(Y.T, V)
        C = numpy.matmul(Binv, YtV)
        trace_C = numpy.trace(C)
        AA = numpy.matmul(A, A)
        trace_AA = numpy.trace(AA)
        trace_M2 = trace_S2inv - 2.0*trace_C + trace_AA

        # Trace of (KM)**2
        if numpy.abs(sigma) < tol:
            trace_K2 = K_mixed.trace(0, exponent=2)
            D = numpy.matmul(X.T, X)
            K2X = K_mixed.dot(0, X, exponent=2)
            E = numpy.matmul(K2X, D)
            E2 = numpy.matmul(E, E)
            trace_KMKM = (trace_K2 - 2.0*numpy.trace(E) + numpy.trace(E2)) / \
                sigma0**4
        else:
            trace_KMKM = (n-m)/sigma**4 - (2*eta/sigma**2)*trace_M + \
                (eta**2)*trace_M2

        # Trace of K*(M**2)
        if numpy.abs(sigma) < tol:
            YtKY = numpy.matmul(Y.T, K_mixed.dot(0, Y))
            BinvYtKY = numpy.matmul(Binv, YtKY)
            trace_BinvYtKY = numpy.trace(BinvYtKY)
            trace_KM = K_mixed.trace(0)/sigma0**2 - trace_BinvYtKY
            trace_KMM = trace_KM / sigma0**2
        else:
            trace_KMM = trace_M/sigma**2 - eta*trace_M2

        # Compute second derivatives
        der2_sigma0_sigma0 = 0.5 * (trace_M2 - 2.0*zMMMz)
        der2_sigma_sigma = 0.5 * (trace_KMKM - 2.0*zMKMKMz)
        der2_sigma_sigma0 = 0.5 * (trace_KMM - 2.0*zMMKMz)

        # Hessian
        hessian = numpy.array(
                [[der2_sigma_sigma, der2_sigma_sigma0],
                 [der2_sigma_sigma0, der2_sigma0_sigma0]], dtype=float)

        if sign_switch:
            hessian = -hessian

        return hessian

    # =====
    # M dot
    # =====

    @staticmethod
    def M_dot(K_mixed, Binv, Y, sigma, sigma0, z):
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

        :param K_mixed: An object of class :class:`MixedCorrelation` which
            represents the operator :math:`\\mathbf{K} + \\eta \\mathbf{I}`.
        :type K_mixed: gaussian_proc.MixedCorrelation

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
        tol = 1e-8
        if numpy.abs(sigma) < tol:

            # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
            w = z / sigma0**2

        else:
            eta = (sigma0 / sigma)**2
            w = K_mixed.solve(eta, z) / sigma**2

        # Computing Mz
        Ytz = numpy.matmul(Y.T, z)
        Binv_Ytz = numpy.matmul(Binv, Ytz)
        Y_Binv_Ytz = numpy.matmul(Y, Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        return Mz

    # =======================
    # maximize log likelihood
    # =======================

    @staticmethod
    def maximize_log_likelihood(
            z, X, K_mixed,
            tol=1e-3, hyperparam_guess=[0.2, 0.2], method='Nelder-Mead'):
        """
        Maximizing the log-likelihood function over the space of parameters
        sigma and sigma0

        In this function, hyperparam = [sigma, sigma0].
        """

        print('Maximize log likelihood with sigma sigma0 ...')

        # Partial function with minus to make maximization to a minimization
        sign_switch = True
        log_likelihood_partial_func = partial(
                DirectLikelihood.log_likelihood, z, X, K_mixed, sign_switch)

        log_likelihood_jacobian_partial_func = partial(
                DirectLikelihood.log_likelihood_jacobian, z, X, K_mixed,
                sign_switch)

        log_likelihood_hessian_partial_func = partial(
                DirectLikelihood.log_likelihood_hessian, z, X, K_mixed,
                sign_switch)

        # Minimize
        # method = 'Nelder-Mead'
        # method = 'BFGS'         # coverges to wrong local maxima
        # method = 'CG'
        # method = 'Newton-CG'
        # method = 'dogleg'       # requires hessian
        method = 'trust-exact'  # requires hessian
        # method = 'trust-ncg'    # requires hessian
        res = scipy.optimize.minimize(log_likelihood_partial_func,
                                      hyperparam_guess,
                                      method=method, tol=tol,
                                      jac=log_likelihood_jacobian_partial_func,
                                      hess=log_likelihood_hessian_partial_func)

        print(res)

        print('Iter: %d, Eval: %d, Success: %s'
              % (res.nit, res.nfev, res.success))

        # Extract res
        sigma = res.x[0]
        sigma0 = res.x[1]
        eta = (sigma0/sigma)**2
        max_lp = -res.fun

        # Output distionary
        results = {
            'sigma': sigma,
            'sigma0': sigma0,
            'eta': eta,
            'max_lp': max_lp
        }

        return results

    # ===================
    # plot log likelihood
    # ===================

    @staticmethod
    def plot_log_likelihood(z, X, K_mixed, optimal_hyperparam=None):
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
                lp[i, j] = DirectLikelihood.log_likelihood(
                        z, X, K_mixed, False, [sigma[j], sigma0[i]])

        [sigma_mesh, sigma0_mesh] = numpy.meshgrid(sigma, sigma0)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        p = ax.plot_surface(sigma_mesh, sigma0_mesh, lp, linewidth=0,
                            antialiased=False)
        fig.colorbar(p, ax=ax)

        if optimal_hyperparam is not None:
            opt_sigma = optimal_hyperparam[0]
            opt_sigma0 = optimal_hyperparam[1]
            opt_lp = DirectLikelihood.log_likelihood(
                        z, X, K_mixed, False, optimal_hyperparam)
            plt.plot(opt_sigma, opt_sigma0, opt_lp, markersize=5, marker='o',
                     markerfacecolor='red', markeredgecolor='red')
        ax.set_xlabel(r'$\sigma$')
        ax.set_ylabel(r'$\sigma_0$')
        ax.set_title('Log Likelihood function')

        filename = 'log_likelihood'
        save_plot(plt, filename, transparent_background=False, pdf=True)

        plt.show()
