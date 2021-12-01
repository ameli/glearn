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
from .._utilities.plot_utilities import *                    # noqa: F401, F403
from .._utilities.plot_utilities import load_plot_settings, plt, matplotlib, \
        make_axes_locatable


# ====
# plot
# ====

def plot(double_profile_likelihood, result):
    """
    Plot likelihood function and its derivatives.
    """

    _plot_likelihood_versus_scale(double_profile_likelihood, result)


# ============================
# plot likelihood versus scale
# ============================

def _plot_likelihood_versus_scale(double_profile_likelihood, result):
    """
    Plots log likelihood for scale parameters.
    """

    dimension = double_profile_likelihood.cov.mixed_cor.cor.dimension

    if dimension == 1:
        _plot_likelihood_versus_scale_1d(double_profile_likelihood, result)
    elif dimension == 2:
        _plot_likelihood_versus_scale_2d(double_profile_likelihood, result)
    else:
        raise ValueError('Likelihood of only 1 and 2 dimensional cases can ' +
                         'be plotted.')


# ===============================
# plot likelihood versus scale 1d
# ===============================

def _plot_likelihood_versus_scale_1d(double_profile_likelihood, result=None):
    """
    Plots log likelihood versus sigma, eta hyperparam
    """

    load_plot_settings()

    # Generate ell for various distance scales
    scale = numpy.logspace(-3, 2, 200)
    eta = numpy.zeros((scale.size, ), dtype=float)
    ell = numpy.zeros((scale.size, ), dtype=float)
    der1_ell = numpy.zeros((scale.size, ), dtype=float)
    der2_ell = numpy.zeros((scale.size, ), dtype=float)
    der1_ell_numerical = numpy.zeros((scale.size-2, ), dtype=float)
    der2_ell_numerical = numpy.zeros((scale.size-4, ), dtype=float)
    eta_guess = 1e+1
    sign_switch = False

    # The variable on the abscissa to take derivative with respect to it.
    if double_profile_likelihood.use_log_scale:
        scale_x = numpy.log10(scale)
    else:
        scale_x = scale

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 8))

    for j in range(scale.size):
        double_profile_likelihood.cov.set_scale(scale[j])
        ell[j] = double_profile_likelihood.likelihood(
                sign_switch, eta_guess,
                double_profile_likelihood._scale_to_hyperparam(scale[j]))
        der1_ell[j] = double_profile_likelihood.likelihood_jacobian(
                sign_switch, eta_guess,
                double_profile_likelihood._scale_to_hyperparam(scale[j]))[0]
        der2_ell[j] = double_profile_likelihood.likelihood_hessian(
                sign_switch, eta_guess,
                double_profile_likelihood._scale_to_hyperparam(scale[j]))[0, 0]
        eta[j] = double_profile_likelihood._find_optimal_eta(
                scale[j], eta_guess)

    # Numerical derivative of likelihood
    der1_ell_numerical = (ell[2:] - ell[:-2]) / (scale_x[2:] - scale_x[:-2])
    der2_ell_numerical = (der1_ell_numerical[2:] - der1_ell_numerical[:-2]) / \
        (scale_x[3:-1] - scale_x[1:-3])

    # Exclude large eta
    eta[eta > 1e+16] = numpy.nan

    # Find maximum of ell
    max_index = numpy.argmax(ell)
    optimal_scale = scale[max_index]
    optimal_ell = ell[max_index]

    # Plot
    ax[0, 0].plot(scale, ell, color='black',
                  label=r'$\ell(\hat{\eta}, \theta)$')
    ax[1, 0].plot(scale, der1_ell, color='black', label='Analytic')
    ax[1, 1].plot(scale, der2_ell, color='black', label='Analytic')
    ax[1, 0].plot(scale[1:-1], der1_ell_numerical, '--', color='black',
                  label='Numerical')
    ax[1, 1].plot(scale[2:-2], der2_ell_numerical, '--', color='black',
                  label='Numerical')
    ax[0, 1].plot(scale, eta, color='black', label=r'$\hat{\eta}(\theta)$')
    ax[0, 0].plot(optimal_scale, optimal_ell, 'o', color='black',
                  markersize=4, label=r'$\hat{\theta}$ (brute force)')

    if result is not None:
        opt_scale = result['hyperparam']['scale']
        opt_ell = result['optimization']['max_posterior']
        ax[0, 0].plot(opt_scale, opt_ell, 'o', color='maroon', markersize=4,
                      label=r'$\hat{\theta}$ (optimized)')

    # Plot annotations
    ax[0, 0].legend(loc='lower right')
    ax[0, 1].legend(loc='upper right')
    ax[1, 0].legend(loc='lower right')
    ax[1, 1].legend(loc='lower right')
    ax[0, 0].set_xscale('log')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_yscale('log')
    ax[1, 0].set_xscale('log')
    ax[1, 1].set_xscale('log')
    ax[0, 0].set_xlim([scale[0], scale[-1]])
    ax[0, 1].set_xlim([scale[0], scale[-1]])
    ax[1, 0].set_xlim([scale[0], scale[-1]])
    ax[1, 1].set_xlim([scale[0], scale[-1]])
    ax[0, 1].set_ylim(bottom=0.0, top=None)
    ax[0, 0].set_xlabel(r'$\theta$')
    ax[0, 1].set_xlabel(r'$\theta$')
    ax[1, 0].set_xlabel(r'$\theta$')
    ax[1, 1].set_xlabel(r'$\theta$')
    ax[0, 0].set_ylabel(r'$\ell(\hat{\eta}(\theta), \theta)$')
    if double_profile_likelihood.use_log_scale:
        ax[1, 0].set_ylabel(
            r'$\frac{\mathrm{d}\ell(\hat{\eta}(\theta),' +
            r' \theta)}{\mathrm{d} (\ln \theta)}$')
    else:
        ax[1, 0].set_ylabel(
            r'$\frac{\mathrm{d}\ell(\hat{\eta}(\theta),' +
            r' \theta)}{\mathrm{d} \theta}$')
    if double_profile_likelihood.use_log_scale:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2\ell(\hat{\eta}(\theta),' +
            r' \theta)}{\mathrm{d} (\ln \theta)^2}$')
    else:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\hat{\eta}(\theta),' +
            r' \theta)}{\mathrm{d} \theta}^2$')
    ax[0, 1].set_ylabel(r'$\hat{\eta}(\theta)$')
    ax[0, 0].set_title(r'Log likelihood function profiled for $\eta$')
    ax[0, 1].set_title(r'Optimal $\eta$')
    ax[1, 0].set_title(r'First derivative of log likelihood function')
    ax[1, 1].set_title(r'Second derivative of log likelihood function')
    ax[0, 0].grid(True)
    ax[0, 1].grid(True)
    ax[1, 0].grid(True)
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


# ===============================
# plot likelihood versus scale 2d
# ===============================

def _plot_likelihood_versus_scale_2d(double_profile_likelihood, result=None):
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
    eta_guess = 1.0
    sign_switch = False

    for i in range(scale2.size):
        for j in range(scale1.size):
            scale = [scale1[j], scale2[i]]
            double_profile_likelihood.cov.set_scale(scale)
            ell[i, j] = double_profile_likelihood.likelihood(
                    sign_switch, eta_guess,
                    double_profile_likelihood._scale_to_hyperparam(scale))
            eta[i, j] = double_profile_likelihood._find_optimal_eta(
                    scale, eta_guess)

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

    # Contour fill Plot of likelihood versus scales
    levels1 = numpy.linspace(min_z, max_z, 2000)
    c1 = ax[0].contourf(x, y, z, levels1, cmap=colormap, zorder=-9)
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(c1, cax=cax1, orientation='vertical')
    cbar1.ax.set_ylabel(r'$\ell(\hat{\eta}(\theta_1, \theta_2), ' +
                        r'\theta_1, \theta_2)$')
    # c.set_clim(0, clim)
    # cbar.set_ticks([0,0.3,0.6,0.9,1])

    # Contour fill Plot of optimal eta versus scales
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
