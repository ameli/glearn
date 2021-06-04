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
from scipy.optimize import minimize
from functools import partial

from .._utilities.plot_utilities import *                    # noqa: F401, F403
from .._utilities.plot_utilities import load_plot_settings, save_plot, plt, \
        mark_inset, InsetPosition
from ._root_finding import find_interval_with_sign_change, chandrupatla_method


# ==================
# Profile Likelihood
# ==================

class ProfileLikelihood(object):

    # ==============
    # Log Likelihood
    # ==============

    def log_likelihood(z, X, K_mixed, sign_switch, hyperparam):
        """
        Log likelihood function

            L = -(1/2) log det(S) - (1/2) log det(X.T*Sinv*X) -
                (1/2) sigma^(-2) * z.T * M1 * z

        where
            S = sigma^2 Kn is the covariance
            Sinv is the inverse of S
            M1 = Sinv = Sinv*X*(X.T*Sinv*X)^(-1)*X.T*Sinv

        hyperparam = [sigma, eta]

        sign_switch chnages the sign of the output from lp to -lp. When True,
        this is used to minimizing (instad of maximizing) the negative of
        log-likelihood function.
        """

        # hyperparam
        sigma = hyperparam[0]
        eta = hyperparam[1]

        logdet_Kn = K_mixed.logdet(eta)

        # Compute log det (X.T Kn_inv X)
        n, m = X.shape
        Y = K_mixed.solve(eta, X)
        w = K_mixed.solve(eta, z)

        XtKninvX = numpy.matmul(X.T, Y)
        logdet_XtKninvX = numpy.log(numpy.linalg.det(XtKninvX))

        # Suppose B is XtKninvX found above. We compute inverse of B
        Binv = numpy.linalg.inv(XtKninvX)
        YBinvYt = numpy.matmul(Y, numpy.matmul(Binv, Y.T))

        # Log likelihood
        lp = -0.5*(n-m)*numpy.log(sigma**2) - 0.5*logdet_Kn \
            - 0.5*logdet_XtKninvX \
            - (0.5/(sigma**2))*numpy.dot(z, w-numpy.dot(YBinvYt, z))

        # If lp is used in scipy.optimize.minimize, change the sign to optain
        # the minimum of -lp
        if sign_switch:
            lp = -lp

        return lp

    # =======================
    # log likelihood der1 eta
    # =======================

    def log_likelihood_der1_eta(z, X, K_mixed, log_eta):
        """
        lp is the log likelihood probability. lp_deta is d(lp)/d(eta), is the
        derivative of lp with respect to eta when the optimal value of sigma is
        subtituted in the likelihood function per given eta.
        """

        # Change log_eta to eta
        if numpy.isneginf(log_eta):
            eta = 0.0
        else:
            eta = 10.0**log_eta

        # Compute Kn_inv*X and Kn_inv*z
        Y = K_mixed.solve(eta, X)
        w = K_mixed.solve(eta, z)

        n, m = X.shape

        # Splitting M into M1 and M2. Here, we compute M2
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)
        Ytz = numpy.matmul(Y.T, z)
        Binv_Ytz = numpy.matmul(Binv, Ytz)
        Y_Binv_Ytz = numpy.matmul(Y, Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        # Traces
        trace_Kninv = K_mixed.traceinv(eta)
        YtY = numpy.matmul(Y.T, Y)
        TraceBinvYtY = numpy.trace(numpy.matmul(Binv, YtY))
        # TraceBinvYtY = numpy.trace(numpy.matmul(Y, numpy.matmul(Binv, Y.T)))
        TraceM = trace_Kninv - TraceBinvYtY

        # Derivative of log likelihood
        zMz = numpy.dot(z, Mz)
        zM2z = numpy.dot(Mz, Mz)
        sigma02 = zMz/(n-m)
        # dlp_deta = -0.5*((TraceM/(n-m))*zMz - zM2z)
        dlp_deta = -0.5*(TraceM - zM2z/sigma02)

        return dlp_deta

    # =======================
    # log likelihood der2 eta
    # =======================

    @staticmethod
    def log_likelihood_der2_eta(z, X, K_mixed, eta):
        """
        The second derivative of lp is computed as a function of only eta.
        Here, we substituted optimal value of sigma, which istself is a
        function of eta.
        """

        Y = K_mixed.solve(eta, X)
        V = K_mixed.solve(eta, Y)
        w = K_mixed.solve(eta, z)

        n, m = X.shape

        # Splitting M
        B = numpy.matmul(X.T, Y)
        Binv = numpy.linalg.inv(B)
        Ytz = numpy.matmul(Y.T, z)
        Binv_Ytz = numpy.matmul(Binv, Ytz)
        Y_Binv_Ytz = numpy.matmul(Y, Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        # Trace of M
        trace_Kninv = K_mixed.traceinv(eta)
        YtY = numpy.matmul(Y.T, Y)
        A = numpy.matmul(Binv, YtY)
        trace_A = numpy.trace(A)
        trace_M = trace_Kninv - trace_A

        # Trace of M**2
        trace_Kn2inv = K_mixed.traceinv(eta, exponent=2)
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
        v = K_mixed.solve(eta, Mz)
        MMz = v - Y_Binv_YtMz

        # Second derivative (only at the location ofzero first derivative)
        zMz = numpy.dot(z, Mz)
        # zM2z = numpy.dot(Mz, Mz)
        zM3z = numpy.dot(Mz, MMz)
        sigma02 = zMz / (n-m)
        # d2lp_deta2 = 0.5*(trace_M2 * zM2z - 2.0*trace_M * zM3z)
        d2lp_deta2 = (0.5/sigma02) * \
            ((trace_M2/(n-m) + (trace_M/(n-m))**2) * zMz - 2.0*zM3z)

        return d2lp_deta2

    # ======================================
    # maximize log likelihood with sigma eta
    # ======================================

    @staticmethod
    def maximize_log_likelihood_with_sigma_eta(
            z, X, K_mixed,
            tol=1e-6, hyperparam_guess=[0.1, 0.1], method='Nelder-Mead'):
        """
        Maximizing the log-likelihood function over the space of
        hyperparam sigma and eta.
        """

        print('Maximize log likelihood with sigma eta ...')

        # Partial function with minus to make maximization to a minimization
        sign_switch = True
        log_likelihood_partial_function = partial(
                ProfileLikelihood.log_likelihood, z, X, K_mixed, sign_switch)

        # Minimize
        # method = 'BFGS'
        # method = 'CG'
        method = 'Nelder-Mead'
        res = minimize(log_likelihood_partial_function, hyperparam_guess,
                       method=method, tol=tol)

        print('Iter: %d, Eval: %d, success: %s'
              % (res.nit, res.nfev, res.success))

        # Extract results
        sigma = res.x[0]
        eta = res.x[1]
        sigma0 = numpy.sqrt(eta) * sigma
        max_lp = -res.fun

        # Output distionary
        results = {
            'sigma': sigma,
            'sigma0': sigma0,
            'eta': eta,
            'max_lp': max_lp
        }

        return results

    # ==============================
    # find log likelihood der1 zeros
    # ==============================

    def find_log_likelihood_der1_zeros(z, X, K_mixed, interval_eta, tol=1e-6,
                                       max_iterations=100,
                                       num_bracket_trials=3):
        """
        root finding of the derivative of lp.

        The log likelihood function is implicitly a function of eta. We have
        substituted the value of optimal sigma, which itself is a function of
        eta.
        """

        # ------------------
        # find optimal sigma
        # ------------------

        def find_optimal_sigma(z, X, K_mixed, eta):
            """
            Based on a given eta, finds optimal sigma
            """

            Y = K_mixed.solve(eta, X)
            w = K_mixed.solve(eta, z)

            n, m = X.shape
            B = numpy.matmul(X.T, Y)
            Binv = numpy.linalg.inv(B)
            Ytz = numpy.matmul(Y.T, z)
            v = numpy.matmul(Y, numpy.matmul(Binv, Ytz))
            sigma2 = numpy.dot(z, w-v) / (n-m)
            sigma = numpy.sqrt(sigma2)

            return sigma

        # -------------------
        # find optimal sigma0
        # -------------------

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

        # -----------------

        print('Find root of log likelihood derivative ...')

        # Find an interval that the function changes sign before finding its
        # root (known as bracketing the function)
        log_eta_start = numpy.log10(interval_eta[0])
        log_eta_end = numpy.log10(interval_eta[1])

        # Partial function with minus to make maximization to a minimization
        log_likelihood_der1_eta_partial_function = partial(
                ProfileLikelihood.log_likelihood_der1_eta, z, X, K_mixed)

        # Initial points
        bracket = [log_eta_start, log_eta_end]
        bracket_found, bracket, bracket_values = \
            find_interval_with_sign_change(
                    log_likelihood_der1_eta_partial_function, bracket,
                    num_bracket_trials, args=(), )

        if bracket_found:
            # There is a sign change in the interval of eta. Find root of lp
            # derivative

            # Find roots using Brent method
            # method = 'brentq'
            # res = scipy.optimize.root_scalar(
            #         log_likelihood_der1_eta_partial_function,
            #         bracket=bracket,
            #         method=method, xtol=tol)
            # print('Iter: %d, Eval: %d, Converged: %s'
            #         % (res.iterations, res.function_calls, res.converged))

            # Find roots using Chandraputala method
            res = chandrupatla_method(log_likelihood_der1_eta_partial_function,
                                      bracket, bracket_values, verbose=False,
                                      eps_m=tol, eps_a=tol,
                                      maxiter=max_iterations)
            print('Iter: %d' % (res['iterations']))

            # Extract results
            # eta = 10**res.root                       # Use with Brent
            eta = 10**res['root']                      # Use with Chandrupatla
            sigma = find_optimal_sigma(z, X, K_mixed, eta)
            sigma0 = numpy.sqrt(eta) * sigma

            # Check second derivative
            success = True
            # d2lp_deta2 = ProfileLikelihood.log_likelihood_der2_eta(
            #         z, X, K_mixed, eta)
            # if d2lp_deta2 < 0:
            #     success = True
            # else:
            #     success = False

        else:
            # bracket with sign change was not found.

            # Evaluate the function in intervals
            eta_left = bracket[0]
            eta_right = bracket[1]
            dlp_deta_left = bracket_values[0]
            dlp_deta_right = bracket_values[1]

            # Second derivative of log likelihood at eta = zero, using either
            # of the two methods below:
            eta_zero = 0.0
            # method 1: directly from analytical equation
            d2lp_deta2_zero_eta = ProfileLikelihood.log_likelihood_der2_eta(
                    z, X, K_mixed, eta_zero)

            # method 2: usng forward differencing from first derivative
            # dlp_deta_zero_eta = ProfileLikelihood.log_likelihood_der1_eta(
            #         z, X, K_mixed, numpy.log10(eta_zero))
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

            # Find sigma and sigma0
            if eta == 0:
                sigma0 = 0
                sigma = find_optimal_sigma(z, X, K_mixed, eta)
                success = True
            elif eta == numpy.inf:
                sigma = 0
                sigma0 = find_optimal_sigma0(z, X)
                success = True
            else:
                raise ValueError('eta must be zero or inf at this point.')

        # Output distionary
        results = {
            'sigma': sigma,
            'sigma0': sigma0,
            'eta': eta,
            'success': success
        }

        return results

    # ===================
    # plot log likelihood
    # ===================

    @staticmethod
    def plot_log_likelihood(z, X, K_mixed):
        """
        Plots log likelihood versus sigma, eta hyperparam
        """

        eta = numpy.logspace(-3, 3, 20)
        sigma = numpy.logspace(-1, 0, 20)
        lp = numpy.zeros((eta.size, sigma.size))
        for i in range(eta.size):
            for j in range(sigma.size):
                lp[i, j] = ProfileLikelihood.log_likelihood(
                        z, X, K_mixed, False, [sigma[j], eta[i]])

        [sigma_mesh, eta_mesh] = numpy.meshgrid(sigma, eta)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # p = ax.plot_surface(sigma_mesh, eta_mesh, lp, linewidth=0,
        #                     antialiased=False)
        p = ax.plot_surface(numpy.log10(sigma_mesh), numpy.log10(eta_mesh), lp,
                            linewidth=0, antialiased=False)
        fig.colorbar(p, ax=ax)
        # ax.xaxis.set_scale('log')
        # ax.yaxis.set_scale('log')
        # plt.yscale('log')
        ax.set_xlabel(r'$\sigma$')
        ax.set_ylabel(r'$\eta$')
        ax.set_title('Log Likelihood function')
        plt.show

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

    # ============================
    # plot log likelihood der1 eta
    # ============================

    @staticmethod
    def plot_log_likelihood_der1_eta(z, X, K, K_mixed, optimal_eta):
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
            dlp_deta[i] = ProfileLikelihood.log_likelihood_der1_eta(
                    z, X, K_mixed, numpy.log10(eta[i]))

        # Compute upper and lower bound of derivative
        dlp_deta_upper_bound, dlp_deta_lower_bound = \
            ProfileLikelihood.compute_bounds_der1_eta(X, K, eta)

        # Compute asymptote of first derivative, using both first and second
        # order approximation
        try:
            # eta_high_res migh not be defined, depending on plot_optimal_eta
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

            # Avoid inset mark lines interset inset axes by setting its anchor
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
        filename = 'log_likelihood_first_derivative'
        save_plot(plt, filename, transparent_background=False, pdf=True)

        plt.show()
