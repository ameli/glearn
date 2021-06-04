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

        # S is the (sigma**2) * K + (sigma0**2) * I, but we don't construct it
        # Also, Kn is K + eta I, where eta = (sigma0 / sigma)**2
        tol = 1e-8
        if numpy.abs(sigma) < tol:

            # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
            logdet_S = K_mixed.get_matrix_size() * numpy.log(sigma0**2)

            Y = X / sigma**2
            w = z / sigma**2

        else:
            eta = (sigma0 / sigma)**2
            logdet_Kn = K_mixed.logdet(eta)
            logdet_S = K_mixed.get_matrix_size() * numpy.log(sigma**2) + \
                logdet_Kn

            Y = K_mixed.solve(eta, X) / sigma**2
            w = K_mixed.solve(eta, z) / sigma**2

        # Compute log det (X.T*Sinv*X)
        XtSinvX = numpy.matmul(X.T, Y)
        logdet_XtSinvX = numpy.log(numpy.linalg.det(XtSinvX))

        # Matrix B is X.T * S * X
        Binv = numpy.linalg.inv(XtSinvX)
        YBinvYt = numpy.matmul(Y, numpy.matmul(Binv, Y.T))

        # Log likelihood
        lp = -0.5*logdet_S - 0.5*logdet_XtSinvX - \
            0.5*numpy.dot(z, w-numpy.dot(YBinvYt, z))

        # If lp is used in scipy.optimize.minimize, change the sign to optain
        # the minimum of -lp
        if sign_switch:
            lp = -lp

        return lp

    # =======================
    # log likelihood jacobian
    # =======================

    def log_likelihood_jacobian(z, X, K_mixed, sign_switch, hyperparam):
        """
        """

        # hyperparameters
        sigma = hyperparam[0]
        sigma0 = hyperparam[1]
        n, m = X.shape

        # S is the (sigma**2) * K + (sigma0**2) * I, but we don't construct it
        # Also, Kn is K + eta I, where eta = (sigma0 / sigma)**2

        # Computing Y=Sinv*X and w=Sinv*z
        tol = 1e-8
        if numpy.abs(sigma) < tol:

            # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
            Y = X / sigma**2
            w = z / sigma**2

        else:
            eta = (sigma0 / sigma)**2
            Y = K_mixed.solve(eta, X) / sigma**2
            w = K_mixed.solve(eta, z) / sigma**2

        # Computing Mz
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)
        Ytz = numpy.matmul(Y.T, z)
        Binv_Ytz = numpy.matmul(Binv, Ytz)
        Y_Binv_Ytz = numpy.matmul(Y, Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        # Computing KMz
        KMz = K_mixed.dot(0, Mz)

        # Compute Sinv * KMz
        if numpy.abs(sigma) < tol:

            # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
            w2 = KMz / sigma**2

        else:
            w2 = K_mixed.solve(eta, KMz) / sigma**2

        # Compute MKMz
        Yt_KMz = numpy.matmul(Y.T, KMz)
        Binv_Yt_KMz = numpy.matmul(Binv, Yt_KMz)
        Y_Binv_Yt_KMz = numpy.matmul(Y, Binv_Yt_KMz)
        MKMz = w2 - Y_Binv_Yt_KMz

        # Compute zMKMz and zMMz
        zMKMz = numpy.dot(z, MKMz)
        zMMz = numpy.dot(Mz, Mz)

        # Compute trace of KM which is (n-m)/sigma**2 - eta* trace(M)
        trace_Kninv = K_mixed.traceinv(eta)
        YtY = numpy.matmul(Y.T, Y)
        TraceBinvYtY = numpy.trace(numpy.matmul(Binv, YtY))
        # TraceBinvYtY = numpy.trace(numpy.matmul(Y, numpy.matmul(Binv, Y.T)))
        traceM = trace_Kninv - TraceBinvYtY
        traceKM = (n - m) / sigma**2 - eta * traceM

        # Derivative of lp wrt to sigma
        der1_sigma = -0.5*traceKM + 0.5*zMKMz
        der1_sigma0 = -0.5*traceM + 0.5*zMMz

        if sign_switch:
            der1_sigma = -der1_sigma
            der1_sigma0 = -der1_sigma0

        jacobian = [der1_sigma, der1_sigma0]

        return jacobian

    # =======================
    # maximize log likelihood
    # =======================

    @staticmethod
    def maximize_log_likelihood(
            z, X, K_mixed,
            tol=1e-3, hyperparam_guess=[0.1, 0.15], method='Neldeer-Mead'):
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

        # Minimize
        # method = 'BFGS'
        method = 'Newton-CG'
        # method = 'CG'
        # method = 'Nelder-Mead'
        res = scipy.optimize.minimize(log_likelihood_partial_func,
                                      hyperparam_guess,
                                      method=method, tol=tol,
                                      jac=log_likelihood_jacobian_partial_func)

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
