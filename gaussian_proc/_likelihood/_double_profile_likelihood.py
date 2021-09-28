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
    """
    Likelihood function that is profiled with respect to :math:`\\sigma` and
    :math:`\\eta` variables.
    """

    # ====
    # init
    # ====

    def __init__(self, z, X, cov):
        """
        Initialization
        """

        # Attributes
        self.z = z
        self.X = X
        self.cov = cov
        self.mixed_cor = self.cov.mixed_cor
        self.profile_likelihood = ProfileLikelihood(z, X, cov)

        # Configuration
        self.hyperparam_tol = 1e-8
        self.use_logscale = True

        # Store ell, its Jacobian and Hessian.
        self.ell = None
        self.ell_jacobian = None
        self.ell_hessian = None

        # Store hyperparam used to compute ell, its Jacobian and Hessian.
        self.ell_hyperparam = None
        self.ell_jacobian_hyperparam = None
        self.ell_hessian_hyperparam = None

    # ==========
    # likelihood
    # ==========

    def likelihood(
            self,
            sign_switch,
            log_eta_guess,
            hyperparam):
        """
        Variable eta is profiled out, meaning that optimal value of eta is
        used in log-likelihood function.
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

        # Here, hyperparam consists of only scale, but not eta.
        if isinstance(hyperparam, list):
            hyperparam = numpy.array(hyperparam)
        scale = numpy.abs(hyperparam)
        self.cov.set_scale(scale)

        # Find optimal eta
        eta = self._find_optimal_eta(scale, log_eta_guess)
        log_eta = numpy.log(eta)

        # Construct new hyperparam that consists of both eta and scale
        hyperparam_full = numpy.r_[log_eta, scale]

        # Finding the maxima
        ell = self.profile_likelihood.likelihood(sign_switch, hyperparam_full)

        # Store ell to member data (without sign-switch).
        self.ell = ell
        self.ell_hyperparam = hyperparam

        return ell

    # ===================
    # likelihood jacobian
    # ===================

    def likelihood_jacobian(
            self,
            sign_switch,
            log_eta_guess,
            hyperparam):
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

        # When profiling eta is enabled, derivative w.r.t eta is not needed.
        # Compute only Jacobian w.r.t scale. Also, here, the input hyperparam
        # consists of only scale (and not eta).
        if isinstance(hyperparam, list):
            hyperparam = numpy.array(hyperparam)
        scale = numpy.abs(hyperparam)
        self.cov.set_scale(scale)

        # Find optimal eta
        eta = self._find_optimal_eta(scale, log_eta_guess)
        log_eta = numpy.log(eta)

        # Construct new hyperparam that consists of both eta and scale
        hyperparam_full = numpy.r_[log_eta, scale]

        # Compute first derivative w.r.t scale
        der1_scale = self.profile_likelihood._likelihood_der1_scale(
                hyperparam_full)

        # Jacobian only consists of the derivative w.r.t scale
        jacobian = der1_scale

        # Store jacobian to member data (without sign-switch).
        self.ell_jacobian = jacobian
        self.ell_jacobian_hyperparam = hyperparam

        if sign_switch:
            jacobian = -jacobian

        return jacobian

    # ==================
    # likelihood hessian
    # ==================

    def likelihood_hessian(self, sign_switch, log_eta_guess, hyperparam):
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

        # When profiling eta is enabled, derivative w.r.t eta is not needed.
        # Compute only Jacobian w.r.t scale. Also, here, the input hyperparam
        # consists of only scale (and not eta).
        if isinstance(hyperparam, list):
            hyperparam = numpy.array(hyperparam)
        scale = numpy.abs(hyperparam)
        self.cov.set_scale(scale)

        # Find optimal eta
        eta = self._find_optimal_eta(scale, log_eta_guess)
        log_eta = numpy.log(eta)

        # Construct new hyperparam that consists of both eta and scale
        hyperparam_full = numpy.r_[log_eta, scale]

        # Compute second derivative w.r.t scale
        der2_scale = self.profile_likelihood._likelihood_der2_scale(
                hyperparam_full)

        # Concatenate derivatives to form Hessian of all variables
        hessian = der2_scale

        # Store hessian to member data (without sign-switch).
        self.ell_hessian = hessian
        self.ell_hessian_hyperparam = hyperparam

        if sign_switch:
            hessian = -hessian

        return hessian

    # ================
    # find optimal eta
    # ================

    def _find_optimal_eta(
            self,
            scale,
            log_eta_guess=0.0,
            optimization_method='Nelder-Mead'):
        """
        Finds optimal eta to profile it out of the log-likelihood.
        """

        self.cov.set_scale(scale)

        # # Note: When using interpolation, make sure the interval below is
        # # exactly the end points of eta_i, not less or more.
        # min_eta_guess = numpy.min([1e-4, 10.0**log_eta_guess * 1e-2])
        # max_eta_guess = numpy.max([1e+3, 10.0**log_eta_guess * 1e+2])
        # interval_eta = [min_eta_guess, max_eta_guess]
        #
        # # Using root finding method on the first derivative w.r.t eta
        # result = self.profile_likelihood.find_likelihood_der1_zeros(
        #         interval_eta)
        # eta = result['hyperparam']['eta']

        # optimization_method = 'Newton-CG'
        result = self.profile_likelihood.maximize_likelihood(
                tol=1e-3, hyperparam_guess=[log_eta_guess],
                optimization_method=optimization_method)

        eta = result['hyperparam']['eta']

        return eta

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

        # When profile eta is used, hyperparam only contains scale
        log_eta_guess = 1.0

        # Partial function of likelihood with profiled eta. The input
        # hyperparam is only scale, not eta.
        sign_switch = True
        likelihood_partial_func = partial(self.likelihood, sign_switch,
                                          log_eta_guess)

        # Partial function of Jacobian of likelihood (with minus sign)
        jacobian_partial_func = partial(self.likelihood_jacobian, sign_switch,
                                        log_eta_guess)

        # Partial function of Hessian of likelihood (with minus sign)
        hessian_partial_func = partial(self.likelihood_hessian, sign_switch,
                                       log_eta_guess)

        # Minimize
        res = scipy.optimize.minimize(
                likelihood_partial_func, hyperparam_guess,
                method=optimization_method, tol=tol, jac=jacobian_partial_func,
                hess=hessian_partial_func)

        # Get the optimal scale
        scale = numpy.abs(res.x)

        # Find optimal eta with the given scale
        eta = self._find_optimal_eta(scale, log_eta_guess)

        # Find optimal sigma and sigma0 with the optimal eta
        sigma, sigma0 = self.profile_likelihood._find_optimal_sigma_sigma0(eta)
        max_ell = -res.fun

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
                'scale': scale,
            },
            'optimization':
            {
                'max_likelihood': max_ell,
                'iter': res.nit,
            },
            'time':
            {
                'wall_time': wall_time,
                'proc_time': proc_time
            }
        }

        return result

    # ============================
    # plot likelihood versus scale
    # ============================

    def plot_likelihood_versus_scale(self, result):
        """
        Plots log likelihood for scale parameters.
        """

        dimension = self.cov.mixed_cor.cor.dimension

        if dimension == 1:
            self.plot_likelihood_versus_scale_1d(result)
        elif dimension == 2:
            self.plot_likelihood_versus_scale_2d(result)
        else:
            raise ValueError('Likelihood of only 1 and 2 dimensional cases ' +
                             'can be plotted.')

    # ===============================
    # plot likelihood versus scale 1d
    # ===============================

    def plot_likelihood_versus_scale_1d(self, result=None):
        """
        Plots log likelihood versus sigma, eta hyperparam
        """

        load_plot_settings()

        # Generate ell for various distance scales
        scale = numpy.logspace(-3, 2, 200)
        eta = numpy.zeros((scale.size, ), dtype=float)
        ell = numpy.zeros((scale.size, ), dtype=float)
        der1_ell = numpy.zeros((scale.size, ), dtype=float)
        der1_ell_numerical = numpy.zeros((scale.size-2, ), dtype=float)
        log_eta_guess = 1.0
        sign_switch = False

        fig, ax = plt.subplots(ncols=2, figsize=(9, 4.5))
        ax2 = ax[0].twinx()

        for j in range(scale.size):
            self.cov.set_scale(scale[j])
            ell[j] = self.likelihood(sign_switch, log_eta_guess, scale[j])
            der1_ell[j] = self.likelihood_jacobian(sign_switch, log_eta_guess,
                                                   scale[j])[0]
            eta[j] = self._find_optimal_eta(scale[j], log_eta_guess)

        # Numerical derivative of likelihood
        der1_ell_numerical = (ell[2:] - ell[:-2]) / (scale[2:] - scale[:-2])

        # Exclude large eta
        eta[eta > 1e+16] = numpy.nan

        # Find maximum of ell
        max_index = numpy.argmax(ell)
        optimal_scale = scale[max_index]
        optimal_ell = ell[max_index]

        # Plot
        ax[0].plot(scale, ell, color='black',
                   label=r'$\ell(\hat{\eta}, \theta)$')
        ax[1].plot(scale, der1_ell, color='black', label='Analytic')
        ax[1].plot(scale[1:-1], der1_ell_numerical, '--', color='black',
                   label='Numerical')
        ax2.plot(scale, eta, '--', color='black',
                 label=r'$\hat{\eta}(\theta)$')
        ax[0].plot(optimal_scale, optimal_ell, 'o', color='black',
                   markersize=4, label=r'$\hat{\theta}$ (brute force)')

        if result is not None:
            opt_scale = result['hyperparam']['scale']
            opt_ell = result['optimization']['max_likelihood']
            ax[0].plot(opt_scale, opt_ell, 'o', color='maroon', markersize=4,
                       label=r'$\hat{\theta}$ (optimized)')

        # Plot annotations
        ax[0].legend(loc='lower right')
        ax[1].legend(loc='lower right')
        ax2.legend(loc='upper right')
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].set_xlim([scale[0], scale[-1]])
        ax[1].set_xlim([scale[0], scale[-1]])
        ax2.set_xlim([scale[0], scale[-1]])
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

    # ===============================
    # plot likelihood versus scale 2d
    # ===============================

    def plot_likelihood_versus_scale_2d(self, result=None):
        """
        Plots log likelihood versus sigma, eta hyperparam
        """

        load_plot_settings()

        # Optimal point
        optimal_scale = result['hyperparam']['scale']

        # Generate ell for various distance scales
        scale1 = numpy.logspace(-2, 1, 10)
        scale2 = numpy.logspace(-2, 1, 10)
        ell = numpy.zeros((scale2.size, scale1.size), dtype=float)
        eta = numpy.zeros((scale2.size, scale1.size), dtype=float)
        log_eta_guess = 1.0
        sign_switch = False

        for i in range(scale2.size):
            for j in range(scale1.size):
                scale = [scale1[j], scale2[i]]
                self.cov.set_scale(scale)
                ell[i, j] = self.likelihood(sign_switch, log_eta_guess, scale)
                eta[i, j] = self._find_optimal_eta(scale, log_eta_guess)

        # Convert inf to nan
        ell = numpy.where(numpy.isinf(ell), numpy.nan, ell)
        eta = numpy.where(numpy.isinf(eta), numpy.nan, eta)
        eta[eta > 1e+16] = numpy.nan

        # Smooth data for finer plot
        # sigma_ = [2, 2]  # in unit of data pixel size
        # ell = scipy.ndimage.filters.gaussian_filter(
        #         ell, sigma_, mode='nearest')
        # eta = scipy.ndimage.filters.gaussian_filter(
        #         eta, sigma_, mode='nearest')

        # Increase resolution for better contour plot
        N = 300
        f1 = scipy.interpolate.interp2d(
                numpy.log10(scale1), numpy.log10(scale2), ell, kind='cubic')
        f2 = scipy.interpolate.interp2d(
                numpy.log10(scale1), numpy.log10(scale2), eta, kind='cubic')
        scale1_fine = numpy.logspace(
                numpy.log10(scale1[0]), numpy.log10(scale1[-1]), N)
        scale2_fine = numpy.logspace(
                numpy.log10(scale2[0]), numpy.log10(scale2[-1]), N)
        x, y = numpy.meshgrid(scale1_fine, scale2_fine)
        ell_fine = f1(numpy.log10(scale1_fine), numpy.log10(scale2_fine))
        eta_fine = f2(numpy.log10(scale1_fine), numpy.log10(scale2_fine))

        # Find maximum of ell
        max_indices = numpy.unravel_index(numpy.nanargmax(ell_fine),
                                          ell_fine.shape)
        opt_scale1 = scale1_fine[max_indices[1]]
        opt_scale2 = scale2_fine[max_indices[0]]
        # opt_ell = ell_fine[max_indices[0], max_indices[1]]

        # We will plot the difference of max of ell to ell, called z
        # z = max_ell - ell_fine
        z = ell_fine

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
        ax[0].plot(opt_scale1, opt_scale2, 'o', color='red', markersize=6,
                   label=r'$(\hat{\theta}_1, \hat{\theta}_2)$ (by brute ' +
                         r'force on grid)')
        ax[1].plot(opt_scale1, opt_scale2, 'o', color='red', markersize=6,
                   label=r'$(\hat{\theta}_1, \hat{\theta}_2)$ (by brute ' +
                         r'force on grid)')

        # Plot optimal point as found by the profile likelihood method
        ax[0].plot(optimal_scale[0], optimal_scale[1], 'o', color='black',
                   markersize=6, label=r'$(\hat{\theta}_1, \hat{\theta}_2)$ ' +
                                       r'(by optimization')
        ax[1].plot(optimal_scale[0], optimal_scale[1], 'o', color='black',
                   markersize=6, label=r'$(\hat{\theta}_1, \hat{\theta}_2)$ ' +
                                       r'(by optimization')

        # Plot annotations
        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlim([scale1[0], scale1[-1]])
        ax[0].set_ylim([scale2[0], scale2[-1]])
        ax[1].set_xlim([scale1[0], scale1[-1]])
        ax[1].set_ylim([scale2[0], scale2[-1]])
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
