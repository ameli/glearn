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
from .._utilities.plot_utilities import load_plot_settings, plt, matplotlib, \
        make_axes_locatable
from ._profile_likelihood import ProfileLikelihood


# =========================
# Double Profile Likelihood
# =========================

class DoubleProfileLikelihood(object):

    # ==========
    # likelihood
    # ==========

    def likelihood(
            z,
            X,
            cov,
            sign_switch,
            log_eta_guess,
            hyperparam):
        """
        Variable eta is profiled out, meaning that optimal value of eta is
        used in log-likelihood function.
        """

        # Here, hyperparam consists of only distance_scale, but not eta.
        if isinstance(hyperparam, list):
            hyperparam = numpy.array(hyperparam)
        distance_scale = numpy.abs(hyperparam)
        cov.set_distance_scale(distance_scale)

        # Find optimal eta
        eta = DoubleProfileLikelihood._find_optimal_eta(
                z, X, cov, distance_scale, log_eta_guess)
        log_eta = numpy.log(eta)

        # Construct new hyperparam that consists of both eta and distance_scale
        hyperparam_full = numpy.r_[log_eta, distance_scale]

        # Finding the maxima.
        lp = ProfileLikelihood.likelihood(
                z, X, cov.mixed_cor, sign_switch, hyperparam_full)

        return lp

    # ===================
    # likelihood jacobian
    # ===================

    @staticmethod
    def likelihood_jacobian(
            z,
            X,
            cov,
            sign_switch,
            log_eta_guess,
            hyperparam):
        """
        Computes Jacobian w.r.t eta, and if given, distance_scale.
        """

        # When profiling eta is enabled, derivative w.r.t eta is not needed.
        # Compute only Jacobian w.r.t distance_scale. Also, here, the input
        # hyperparam consists of only distance_scale (and not eta).
        if isinstance(hyperparam, list):
            hyperparam = numpy.array(hyperparam)
        distance_scale = numpy.abs(hyperparam)
        cov.set_distance_scale(distance_scale)

        # Find optimal eta
        eta = DoubleProfileLikelihood._find_optimal_eta(
                z, X, cov, distance_scale, log_eta_guess)
        log_eta = numpy.log(eta)

        # Construct new hyperparam that consists of both eta and distance_scale
        hyperparam_full = numpy.r_[log_eta, distance_scale]

        # Compute first derivative w.r.t distance_scale
        der1_distance_scale = \
            ProfileLikelihood.likelihood_der1_distance_scale(
                    z, X, cov, hyperparam_full)

        # Jacobian only consists of the derivative w.r.t distance_scale
        jacobian = der1_distance_scale

        if sign_switch:
            jacobian = -jacobian

        return jacobian

    # ==================
    # likelihood hessian
    # ==================

    @staticmethod
    def likelihood_hessian(z, X, cov, sign_switch, log_eta_guess, hyperparam):
        """
        Computes Hessian w.r.t eta, and if given, distance_scale.
        """

        # When profiling eta is enabled, derivative w.r.t eta is not needed.
        # Compute only Jacobian w.r.t distance_scale. Also, here, the input
        # hyperparam consists of only distance_scale (and not eta).
        if isinstance(hyperparam, list):
            hyperparam = numpy.array(hyperparam)
        distance_scale = numpy.abs(hyperparam)
        cov.set_distance_scale(distance_scale)

        # Find optimal eta
        eta = DoubleProfileLikelihood._find_optimal_eta(
                z, X, cov, distance_scale, log_eta_guess)
        log_eta = numpy.log(eta)

        # Construct new hyperparam that consists of both eta and distance_scale
        hyperparam_full = numpy.r_[log_eta, distance_scale]

        # Compute second derivative w.r.t distance_scale
        der2_distance_scale = \
            ProfileLikelihood.likelihood_der2_distance_scale(
                    z, X, cov, hyperparam_full)

        # Concatenate derivatives to form Hessian of all variables
        hessian = der2_distance_scale

        # if sign_switch:
        #     hessian = -hessian

        return hessian

    # ================
    # find optimal eta
    # ================

    def _find_optimal_eta(
            z,
            X,
            cov,
            distance_scale,
            log_eta_guess=0.0,
            optimization_method='Nelder-Mead'):
        """
        Finds optimal eta to profile it out of the log-likelihood.
        """

        cov.set_distance_scale(distance_scale)

        # # Note: When using interpolation, make sure the interval below is
        # # exactly the end points of eta_i, not less or more.
        # min_eta_guess = numpy.min([1e-4, 10.0**log_eta_guess * 1e-2])
        # max_eta_guess = numpy.max([1e+3, 10.0**log_eta_guess * 1e+2])
        # interval_eta = [min_eta_guess, max_eta_guess]
        #
        # # Using root finding method on the first derivative w.r.t eta
        # result = ProfileLikelihood.find_likelihood_der1_zeros(
        #         z, X, cov.mixed_cor, interval_eta)
        # eta = result['hyperparam']['eta']

        # optimization_method = 'Newton-CG'
        result = ProfileLikelihood.maximize_likelihood(
                z, X, cov,
                tol=1e-3,
                hyperparam_guess=[log_eta_guess],
                optimization_method=optimization_method)

        eta = result['hyperparam']['eta']

        return eta

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

        # Keeping times
        initial_wall_time = time.time()
        initial_proc_time = time.process_time()

        # When profile eta is used, hyperparam only contains distance_scale
        log_eta_guess = 1.0

        # Partial function of likelihood with profiled eta. The input
        # hyperparam is only distance_scale, not eta.
        sign_switch = True
        likelihood_partial_func = partial(
                DoubleProfileLikelihood.likelihood, z, X, cov, sign_switch,
                log_eta_guess)

        # Partial function of Jacobian of likelihood (with minus sign)
        jacobian_partial_func = partial(
                DoubleProfileLikelihood.likelihood_jacobian, z, X, cov,
                sign_switch, log_eta_guess)

        # Partial function of Hessian of likelihood (with minus sign)
        hessian_partial_func = partial(
                DoubleProfileLikelihood.likelihood_hessian, z, X, cov,
                sign_switch, log_eta_guess)

        # Minimize
        res = scipy.optimize.minimize(
                likelihood_partial_func, hyperparam_guess,
                method=optimization_method, tol=tol, jac=jacobian_partial_func,
                hess=hessian_partial_func)

        # Get the optimal distance_scale
        distance_scale = numpy.abs(res.x)

        # Find optimal eta with the given distance_scale
        eta = DoubleProfileLikelihood._find_optimal_eta(
                z, X, cov, distance_scale, log_eta_guess)

        # Find optimal sigma and sigma0 with the optimal eta
        sigma, sigma0 = ProfileLikelihood.find_optimal_sigma_sigma0(
                z, X, cov.mixed_cor, eta)
        max_lp = -res.fun

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

    # =====================================
    # plot likelihood versus distance scale
    # =====================================

    @staticmethod
    def plot_likelihood_versus_distance_scale(
            z,
            X,
            cov,
            result):
        """
        Plots log likelihood for distance_scale parameters.
        """

        dimension = cov.mixed_cor.cor.dimension

        if dimension == 1:
            DoubleProfileLikelihood.plot_likelihood_versus_distance_scale_1d(
                    z, X, cov, result)
        elif dimension == 2:
            DoubleProfileLikelihood.plot_likelihood_versus_distance_scale_2d(
                    z, X, cov, result)
        else:
            raise ValueError('Likelihood of only 1 and 2 dimensional cases ' +
                             'can be plotted.')

    # ========================================
    # plot likelihood versus distance scale 1d
    # ========================================

    @staticmethod
    def plot_likelihood_versus_distance_scale_1d(
            z,
            X,
            cov,
            result=None):
        """
        Plots log likelihood versus sigma, eta hyperparam
        """

        load_plot_settings()

        # Generate lp for various distance scales
        distance_scale = numpy.logspace(-3, 2, 200)
        eta = numpy.zeros((distance_scale.size, ), dtype=float)
        lp = numpy.zeros((distance_scale.size, ), dtype=float)
        der1_lp = numpy.zeros((distance_scale.size, ), dtype=float)
        der1_lp_numerical = numpy.zeros((distance_scale.size-2, ), dtype=float)
        log_eta_guess = 1.0
        sign_switch = False

        fig, ax = plt.subplots(ncols=2, figsize=(9, 4.5))
        ax2 = ax[0].twinx()

        for j in range(distance_scale.size):
            cov.set_distance_scale(distance_scale[j])
            lp[j] = DoubleProfileLikelihood.likelihood(
                    z, X, cov, sign_switch, log_eta_guess,
                    distance_scale[j])
            der1_lp[j] = DoubleProfileLikelihood.likelihood_jacobian(
                    z, X, cov, sign_switch, log_eta_guess,
                    distance_scale[j])[0]
            eta[j] = DoubleProfileLikelihood._find_optimal_eta(
                    z, X, cov, distance_scale[j], log_eta_guess)

        # Numerical derivative of likelihood
        der1_lp_numerical = (lp[2:] - lp[:-2]) / \
            (distance_scale[2:] - distance_scale[:-2])

        # Exclude large eta
        eta[eta > 1e+16] = numpy.nan

        # Find maximum of lp
        max_index = numpy.argmax(lp)
        optimal_distance_scale = distance_scale[max_index]
        optimal_lp = lp[max_index]

        # Plot
        ax[0].plot(distance_scale, lp, color='black',
                   label=r'$\ell(\hat{\eta}, \theta)$')
        ax[1].plot(
            distance_scale, der1_lp, color='black', label='Analytic')
        ax[1].plot(
                distance_scale[1:-1], der1_lp_numerical, '--', color='black',
                label='Numerical')
        ax2.plot(distance_scale, eta, '--', color='black',
                 label=r'$\hat{\eta}(\theta)$')
        ax[0].plot(optimal_distance_scale, optimal_lp, 'o', color='black',
                   markersize=4, label=r'$\hat{\theta}$ (brute force)')

        if result is not None:
            opt_distance_scale = result['hyperparam']['distance_scale']
            opt_lp = result['optimization']['max_likelihood']
            ax[0].plot(opt_distance_scale, opt_lp, 'o', color='maroon',
                       markersize=4, label=r'$\hat{\theta}$ (optimized)')

        # Plot annotations
        ax[0].legend(loc='lower right')
        ax[1].legend(loc='lower right')
        ax2.legend(loc='upper right')
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].set_xlim([distance_scale[0], distance_scale[-1]])
        ax[1].set_xlim([distance_scale[0], distance_scale[-1]])
        ax2.set_xlim([distance_scale[0], distance_scale[-1]])
        ax2.set_ylim(bottom=0.0, top=None)
        ax[0].set_xlabel(r'$\theta$')
        ax[1].set_xlabel(r'$\theta$')
        ax[0].set_ylabel(r'$\ell(\hat{\eta}(\theta), \theta)$')
        ax[1].set_ylabel(
            r'$\frac{\mathrm{d}\ell(\hat{\eta}(\theta),' +
            r' \theta)}{\mathrm{d} \theta}$')
        ax2.set_ylabel(r'$\hat{\eta}(\theta)$')
        ax[0].set_title(r'Log likelihood function profiled for $\eta$')
        ax[1].set_title(r'Derivative of log likelihood function')
        ax[0].grid(True)
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    # ========================================
    # plot likelihood versus distance scale 2d
    # ========================================

    @staticmethod
    def plot_likelihood_versus_distance_scale_2d(
            z,
            X,
            cov,
            result=None):
        """
        Plots log likelihood versus sigma, eta hyperparam
        """

        load_plot_settings()

        # Optimal point
        optimal_distance_scale = result['hyperparam']['distance_scale']

        # Generate lp for various distance scales
        distance_scale1 = numpy.logspace(-2, 1, 10)
        distance_scale2 = numpy.logspace(-2, 1, 10)
        lp = numpy.zeros((distance_scale2.size, distance_scale1.size),
                         dtype=float)
        eta = numpy.zeros((distance_scale2.size, distance_scale1.size),
                          dtype=float)
        log_eta_guess = 1.0
        sign_switch = False

        for i in range(distance_scale2.size):
            for j in range(distance_scale1.size):
                distance_scale = [distance_scale1[j], distance_scale2[i]]
                cov.set_distance_scale(distance_scale)
                lp[i, j] = DoubleProfileLikelihood.likelihood(
                        z, X, cov, sign_switch, log_eta_guess,
                        distance_scale)
                eta[i, j] = DoubleProfileLikelihood._find_optimal_eta(
                        z, X, cov, distance_scale, log_eta_guess)

        # Convert inf to nan
        lp = numpy.where(numpy.isinf(lp), numpy.nan, lp)
        eta = numpy.where(numpy.isinf(eta), numpy.nan, eta)
        eta[eta > 1e+16] = numpy.nan

        # Smooth data for finer plot
        # sigma_ = [2, 2]  # in unit of data pixel size
        # lp = scipy.ndimage.filters.gaussian_filter(
        #         lp, sigma_, mode='nearest')
        # eta = scipy.ndimage.filters.gaussian_filter(
        #         eta, sigma_, mode='nearest')

        # Increase resolution for better contour plot
        N = 300
        f1 = scipy.interpolate.interp2d(
                numpy.log10(distance_scale1),
                numpy.log10(distance_scale2), lp, kind='cubic')
        f2 = scipy.interpolate.interp2d(
                numpy.log10(distance_scale1),
                numpy.log10(distance_scale2), eta, kind='cubic')
        distance_scale1_fine = numpy.logspace(
                numpy.log10(distance_scale1[0]),
                numpy.log10(distance_scale1[-1]), N)
        distance_scale2_fine = numpy.logspace(
                numpy.log10(distance_scale2[0]),
                numpy.log10(distance_scale2[-1]), N)
        x, y = numpy.meshgrid(distance_scale1_fine, distance_scale2_fine)
        lp_fine = f1(numpy.log10(distance_scale1_fine),
                     numpy.log10(distance_scale2_fine))
        eta_fine = f2(numpy.log10(distance_scale1_fine),
                      numpy.log10(distance_scale2_fine))

        # Find maximum of lp
        max_indices = numpy.unravel_index(numpy.nanargmax(lp_fine),
                                          lp_fine.shape)
        opt_distance_scale1 = distance_scale1_fine[max_indices[1]]
        opt_distance_scale2 = distance_scale2_fine[max_indices[0]]
        # opt_lp = lp_fine[max_indices[0], max_indices[1]]

        # We will plot the difference of max of Lp to Lp, called z
        # z = max_lp - lp_fine
        z = lp_fine

        # Cut data
        # cut_data = 0.92
        # clim = 0.87
        # z[z>CutData] = CutData   # Used for plotting data without prior

        # Min and max of data
        min_z = numpy.min(z)
        max_z = numpy.max(z)

        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

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
        levels1 = numpy.linspace(min_z, max_z, 2000)
        c1 = ax[0].contourf(x, y, z, levels1, cmap=colormap, zorder=-9)
        divider1 = make_axes_locatable(ax[0])
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cbar1 = fig.colorbar(c1, cax=cax1, orientation='vertical')
        cbar1.ax.set_ylabel(r'$\ell(\hat{\eta}(\theta_1, \theta_2), ' +
                            r'\theta_1, \theta_2)$')
        # c.set_clim(0, clim)
        # cbar.set_ticks([0,0.3,0.6,0.9,1])

        min_eta = numpy.min(numpy.min(eta))
        max_eta = numpy.max(numpy.max(eta))
        levels2 = numpy.linspace(min_eta, max_eta, 2000)
        c2 = ax[1].contourf(x, y, eta_fine, levels2, cmap=colormap, zorder=-9)
        divider2 = make_axes_locatable(ax[1])
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        cbar2 = fig.colorbar(c2, cax=cax2, orientation='vertical')
        cbar2.ax.set_ylabel(r'$\hat{\eta}(\theta_1, \theta_2)$')
        # c.set_clim(0, clim)
        # cbar.set_ticks([0,0.3,0.6,0.9,1])

        # Plot max of the whole 2D array
        ax[0].plot(opt_distance_scale1, opt_distance_scale2, 'o', color='red',
                   markersize=6, label=r'$(\hat{\theta}_1, \hat{\theta}_2)$ ' +
                                       r'(by brute force on grid)')
        ax[1].plot(opt_distance_scale1, opt_distance_scale2, 'o', color='red',
                   markersize=6, label=r'$(\hat{\theta}_1, \hat{\theta}_2)$ ' +
                                       r'(by brute force on grid)')

        # Plot optimal point as found by the profile likelihood method
        ax[0].plot(optimal_distance_scale[0], optimal_distance_scale[1], 'o',
                   color='black', markersize=6,
                   label=r'$(\hat{\theta}_1, \hat{\theta}_2)$ (by ' +
                         r'optimization')
        ax[1].plot(optimal_distance_scale[0], optimal_distance_scale[1], 'o',
                   color='black', markersize=6,
                   label=r'$(\hat{\theta}_1, \hat{\theta}_2)$ (by ' +
                         r'optimization')

        # Plot annotations
        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlim([distance_scale1[0], distance_scale1[-1]])
        ax[0].set_ylim([distance_scale2[0], distance_scale2[-1]])
        ax[1].set_xlim([distance_scale1[0], distance_scale1[-1]])
        ax[1].set_ylim([distance_scale2[0], distance_scale2[-1]])
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[0].set_xlabel(r'$\log_{10}(\theta_1)$')
        ax[1].set_xlabel(r'$\log_{10}(\theta_1)$')
        ax[0].set_ylabel(r'$\log_{10}(\theta_2)$')
        ax[1].set_ylabel(r'$\log_{10}(\theta_2)$')
        ax[0].set_title(r'Log Likelihood profiled over $\eta$ ')
        ax[1].set_title(r'Optimal $\eta$ for $(\theta_1, \theta_2)$')

        plt.tight_layout()
        plt.show()
