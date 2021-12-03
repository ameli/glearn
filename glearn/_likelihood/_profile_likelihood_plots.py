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
from .._utilities.plot_utilities import load_plot_settings, save_plot, plt, \
        mark_inset, InsetPosition, matplotlib, make_axes_locatable


# ====
# plot
# ====

def plot(profile_likelihood, result):
    """
    Plot likelihood function and its derivatives.
    """

    # Plot log-lp versus scale
    _plot_likelihood_versus_scale(
            profile_likelihood, result, numpy.logspace(-2, 2, 5))

    # Plot log-lp versus eta
    _plot_likelihood_versus_eta(
            profile_likelihood, result, numpy.logspace(-2, 2, 5))

    # Contour Plot of log-lp function
    _plot_likelihood_versus_eta_scale(profile_likelihood, result)

    # Plot first derivative of log likelihood
    _plot_likelihood_der1_eta(profile_likelihood, result)


# ============================
# plot likelihood versus scale
# ============================

def _plot_likelihood_versus_scale(
        profile_likelihood,
        result,
        other_etas=None):
    """
    Plots log likelihood versus sigma, eta hyperparam
    """

    # This function can only plot one dimensional data.
    dimension = profile_likelihood.mixed_cor.cor.dimension
    if dimension != 1:
        raise ValueError('To plot likelihood w.r.t "eta" and "scale", the ' +
                         'dimension of the data points should be one.')

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
            raise TypeError('"other_etas" should be either a scalar, list, ' +
                            'or numpy.ndarray.')

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
    if profile_likelihood.use_log_scale:
        scale_x = numpy.log10(scale)
    else:
        scale_x = scale

    d0_ell = numpy.zeros((stencil_size, etas.size, scale.size), dtype=float)
    d1_ell = numpy.zeros((etas.size, scale.size), dtype=float)
    d2_ell = numpy.zeros((etas.size, scale.size), dtype=float)
    d2_mixed_ell = numpy.zeros((etas.size, scale.size), dtype=float)
    d1_ell_numerical = numpy.zeros((stencil_size, etas.size, scale.size-2),
                                   dtype=float)
    d2_ell_numerical = numpy.zeros((etas.size, scale.size-4), dtype=float)
    d2_mixed_ell_numerical = numpy.zeros((etas.size, scale.size-2),
                                         dtype=float)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    colors = matplotlib.cm.nipy_spectral(numpy.linspace(0, 0.9, etas.size))

    for i in range(etas.size):

        # Stencil to perturb eta
        if profile_likelihood.use_log_eta:
            log_eta = numpy.log10(etas[i])
            d_eta = numpy.max([numpy.abs(log_eta) * 1e-3, 1e-4])
            eta_stencil = 10.0**(
                log_eta + d_eta *
                numpy.arange(-stencil_size//2+1, stencil_size//2+1))
        else:
            d_eta = etas[i] * 1e-3
            eta_stencil = etas[i] + \
                d_eta * numpy.arange(-stencil_size//2+1, stencil_size//2+1)

        for j in range(scale.size):

            # Set the scale
            profile_likelihood.mixed_cor.set_scale(scale[j])

            # Likelihood (first index, center_stencil, means the main etas)
            for k in range(stencil_size):
                d0_ell[k, i, j] = profile_likelihood.likelihood(
                        False,
                        profile_likelihood._eta_to_hyperparam(eta_stencil[k]))

            # First derivative of likelihood w.r.t distance scale
            sign_switch = False
            hyperparam = numpy.r_[
                    profile_likelihood._eta_to_hyperparam(etas[i]),
                    profile_likelihood._scale_to_hyperparam(scale[j])]
            d1_ell[i, j] = profile_likelihood.likelihood_jacobian(
                    sign_switch, hyperparam)[1]

            # Second derivative of likelihood w.r.t distance scale
            hessian_ = profile_likelihood.likelihood_hessian(sign_switch,
                                                             hyperparam)
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
        ax[1, 1].plot(scale, d2_mixed_ell[i, :], color=colors[i], label=label)
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

    # ell at infinity eta
    eta_inf = numpy.inf
    ell_inf = profile_likelihood.likelihood(
            False, profile_likelihood._eta_to_hyperparam(eta_inf))

    ax[0, 0].plot([scale[0], scale[-1]], [ell_inf, ell_inf], '-.',
                  color='black', label=r'$\eta = \infty$')

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

    if profile_likelihood.use_log_scale:
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d} \ell(\theta | \eta)}{\mathrm{d} (\ln\theta)}$')
    else:
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d} \ell(\theta | \eta)}{\mathrm{d} \theta}$')

    if profile_likelihood.use_log_scale:
        ax[1, 0].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} ' +
            r'(\ln\theta)^2}$')
    else:
        ax[1, 0].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} \theta^2}$')

    if profile_likelihood.use_log_scale and profile_likelihood.use_log_eta:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} ' +
            r'(\ln \theta) \mathrm{d} (\ln \eta)}$')
    elif profile_likelihood.use_log_scale:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} ' +
            r'(\ln\theta) \mathrm{d} \eta}$')
    elif profile_likelihood.use_log_eta:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} \theta ' +
            r'\mathrm{d} (\ln \eta)}$')
    else:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\theta | \eta)}{\mathrm{d} \theta ' +
            r'\mathrm{d} \eta}$')

    ax[0, 0].set_title(r'Log likelihood function for fixed $\eta$')
    ax[0, 1].set_title(r'First derivative of log likelihood for fixed $\eta$')
    ax[1, 0].set_title(r'Second derivative of log likelihood for fixed $\eta$')
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

def _plot_likelihood_versus_eta(
        profile_likelihood,
        result,
        other_scales=None):
    """
    Plots log likelihood versus sigma, eta hyperparam
    """

    # This function can only plot one dimensional data.
    dimension = profile_likelihood.mixed_cor.cor.dimension
    if dimension != 1:
        raise ValueError('To plot likelihood w.r.t "eta" and "scale", the ' +
                         'dimension of the data points should be one.')

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
            raise TypeError('"other_scales" should be either a scalar, ' +
                            'list, or numpy.ndarray.')

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
    if profile_likelihood.use_log_eta:
        x_eta = numpy.log10(eta)
    else:
        x_eta = eta

    d0_ell = numpy.zeros((stencil_size, scales.size, eta.size,), dtype=float)
    d1_ell = numpy.zeros((scales.size, eta.size,), dtype=float)
    d2_ell = numpy.zeros((scales.size, eta.size,), dtype=float)
    d2_mixed_ell = numpy.zeros((scales.size, eta.size), dtype=float)
    d1_ell_numerical = numpy.zeros(
            (stencil_size, scales.size, eta.size-2,), dtype=float)
    d2_ell_numerical = numpy.zeros((scales.size, eta.size-4,), dtype=float)
    d2_mixed_ell_numerical = numpy.zeros((scales.size, eta.size-2),
                                         dtype=float)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    colors = matplotlib.cm.nipy_spectral(numpy.linspace(0, 0.9, scales.size))

    for i in range(scales.size):

        # Stencil to perturb scale
        if profile_likelihood.use_log_scale:
            log_scale = numpy.log10(scales[i])
            d_scale = numpy.max([numpy.abs(log_scale) * 1e-3, 1e-4])
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
            profile_likelihood.mixed_cor.set_scale(scale_stencil[k])

            for j in range(eta.size):

                # Likelihood
                d0_ell[k, i, j] = profile_likelihood.likelihood(
                        False, profile_likelihood._eta_to_hyperparam(eta[j]))

                if k == center_stencil:

                    hyperparam = numpy.r_[
                        profile_likelihood._eta_to_hyperparam(eta[j]),
                        profile_likelihood._scale_to_hyperparam(
                            scale_stencil[k])]

                    # First derivative w.r.t eta
                    sign_switch = False
                    d1_ell[i, j] = profile_likelihood.likelihood_jacobian(
                            sign_switch, hyperparam)[0]

                    # Second derivative w.r.t eta
                    hessian_ = profile_likelihood.likelihood_hessian(
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

    # ell at infinity eta
    eta_inf = numpy.inf
    ell_inf = profile_likelihood.likelihood(
            False, profile_likelihood._eta_to_hyperparam(eta_inf))

    ax[0, 0].plot([eta[0], eta[-1]], [ell_inf, ell_inf], '-.',
                  color='black', label=r'$\eta = \infty$')

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

    if profile_likelihood.use_log_eta:
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d}\ell(\eta | \theta)}{\mathrm{d} (\ln \eta)}$')
    else:
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d}\ell(\eta | \theta)}{\mathrm{d}\eta}$')

    if profile_likelihood.use_log_eta:
        ax[1, 0].set_ylabel(
            r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d} ' +
            r'(\ln \eta)^2}$')
    else:
        ax[1, 0].set_ylabel(
            r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d} \eta^2}$')

    if profile_likelihood.use_log_eta and profile_likelihood.use_log_scale:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d} (\ln \eta) ' +
            r'\mathrm{d} (\ln \theta)}$')
    elif profile_likelihood.use_log_eta:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d} (\ln \eta) ' +
            r'\mathrm{d} \theta}$')
    elif profile_likelihood.use_log_scale:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d}\eta ' +
            r'\mathrm{d} (\ln \theta)}$')
    else:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2\ell(\eta | \theta)}{\mathrm{d}\eta ' +
            r'\mathrm{d} \theta}$')

    ax[0, 0].set_title(r'Log likelihood for fixed $\theta$')
    ax[0, 1].set_title(r'First derivative of log likelihood for fixed ' +
                       r'$\theta$')
    ax[1, 0].set_title(r'Second derivative of log likelihood for fixed ' +
                       r'$\theta$')
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

def _plot_likelihood_versus_eta_scale(profile_likelihood, result):
    """
    Plots log likelihood versus sigma and eta hyperparam.
    """

    # This function can only plot one dimensional data.
    dimension = profile_likelihood.mixed_cor.cor.dimension
    if dimension != 1:
        raise ValueError('To plot likelihood w.r.t "eta" and "scale", the ' +
                         'dimension of the data points should be one.')

    load_plot_settings()

    # Optimal point
    optimal_eta = result['hyperparam']['eta']
    optimal_scale = result['hyperparam']['scale']
    optimal_ell = result['optimization']['max_fun']

    eta = numpy.logspace(-3, 3, 50)
    scale = numpy.logspace(-3, 2, 50)
    ell = numpy.zeros((scale.size, eta.size), dtype=float)

    # Compute ell
    for i in range(scale.size):
        profile_likelihood.mixed_cor.set_scale(scale[i])
        for j in range(eta.size):
            ell[i, j] = profile_likelihood.likelihood(
                False, profile_likelihood._eta_to_hyperparam(eta[j]))

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
               label=r'$(\hat{\eta}, \hat{\theta})$ (by brute force on grid)')
    ax[1].plot(opt_eta, opt_ell, 'o', color='red',
               label=r'$\ell(\hat{\eta}, \hat{\theta})$ (by brute force on ' +
                     r'grid)')
    ax[2].plot(opt_scale, opt_ell, 'o', color='red',
               label=r'$\ell(\hat{\eta}, \hat{\theta})$ (by brute force on ' +
                     r'grid)')

    # Plot optimal point as found by the profile likelihood method
    ax[0].plot(optimal_eta, optimal_scale, 'o', color='black', markersize=6,
               label=r'$(\hat{\eta}, \hat{\theta})$ (by optimization)')
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
    ax[0].set_xlabel(r'$\eta$')
    ax[1].set_xlabel(r'$\eta$')
    ax[2].set_xlabel(r'$\theta$')
    ax[0].set_ylabel(r'$\theta$')
    ax[1].set_ylabel(r'$\ell(\eta, \hat{\theta}(\eta))$')
    ax[2].set_ylabel(r'$\ell(\hat{\eta}(\theta), \theta)$')
    ax[0].set_title('Log likelihood function')
    ax[1].set_title(r'Log Likelihood profiled over $\theta$ ')
    ax[2].set_title(r'Log likelihood profiled over $\eta$')
    ax[1].grid(True)
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()


# ========================
# plot likelihood der1 eta
# ========================

def _plot_likelihood_der1_eta(profile_likelihood, result):
    """
    Plots the derivative of log likelihood as a function of eta. Also it shows
    where the optimal eta is, which is the location where the derivative is
    zero.
    """

    load_plot_settings()

    # Optimal point
    optimal_eta = result['hyperparam']['eta']
    optimal_scale = result['hyperparam']['scale']
    profile_likelihood.mixed_cor.set_scale(optimal_scale)

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
            eta = numpy.r_[eta_low_res_left, eta_high_res, eta_low_res_right]

    else:
        eta = numpy.logspace(log_eta_start, log_eta_end, 100)

    # Compute derivative of L
    dell_deta = numpy.zeros(eta.size)
    for i in range(eta.size):
        dell_deta[i] = profile_likelihood._likelihood_der1_eta(
                profile_likelihood._eta_to_hyperparam(eta[i]))

    # Compute upper and lower bound of derivative
    dell_deta_upper_bound, dell_deta_lower_bound = \
        profile_likelihood.bounds_der1_eta(eta)

    # Compute asymptote of first derivative, using both first and second
    # order approximation
    try:
        # eta_high_res might not be defined, depending on plot_optimal_eta
        x = eta_high_res
    except NameError:
        x = numpy.logspace(1, log_eta_end, 100)

    # To compute both first and second order asymptotic relations, compute the
    # second orders first bemuse it also computes the first order.
    asym_maxima_ord2 = profile_likelihood.asymptotic_maxima(degree=2)
    asym_maxima_ord1 = profile_likelihood.asymptotic_maxima(degree=1)
    asym_dell_deta_ord2 = profile_likelihood._asymptotic_likelihood_der1_eta(
            x, degree=2)
    asym_dell_deta_ord1 = profile_likelihood._asymptotic_likelihood_der1_eta(
            x, degree=1)

    # Main plot
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    axes = [ax1]

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
        axes.append(ax2)

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
        ax2.semilogx(x, asym_dell_deta_ord1,
                     label=r'$1^{\text{st}}$ order asymptote',
                     color='chocolate')
        ax2.semilogx(x, asym_dell_deta_ord2,
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

        if asym_maxima_ord1 != []:
            ax2.semilogx(asym_maxima_ord1[-1], 0, marker='o', markersize=6,
                         linewidth=0, color='white',
                         markerfacecolor='chocolate',
                         label=r'Approximated root at $\hat{\eta}_1 = ' +
                               r'10^{%0.2f}$'
                               % numpy.log10(asym_maxima_ord1[-1]))

        if asym_maxima_ord2 != []:
            ax2.semilogx(asym_maxima_ord2[-1], 0, marker='o', markersize=6,
                         linewidth=0, color='white',
                         markerfacecolor='olivedrab',
                         label=r'Approximated root at $\hat{\eta}_2 = ' +
                               r'10^{%0.2f}$'
                               % numpy.log10(asym_maxima_ord2[-1]))

        ax2.set_xlim([eta_high_res[0], eta_high_res[-1]])
        # plt.setp(ax2.get_yticklabels(), backgroundcolor='white')

        # Find suitable range for plot limits
        min_plot = numpy.abs(numpy.min(dell_deta))
        min_plot_base = 10**numpy.floor(numpy.log10(numpy.abs(min_plot)))
        # min_plot_lim = numpy.ceil(min_plot/min_plot_base)*min_plot_base
        min_plot_lim = numpy.ceil(min_plot/min_plot_base + 1.0) * \
            min_plot_base
        ax2.set_ylim([-min_plot_lim, min_plot_lim])
        ax2.set_yticks([-numpy.abs(min_plot_lim), 0, numpy.abs(min_plot_lim)])

        ax2.text(optimal_eta*10**0.05, min_plot_lim*0.05,
                 r'$\hat{\eta}$' % numpy.log10(optimal_eta),
                 horizontalalignment='left', verticalalignment='bottom',
                 fontsize=10)

        for i in range(len(asym_maxima_ord1)):
            ax2.text(asym_maxima_ord1[i]*10**0.05, min_plot_lim*0.05,
                     r'$\hat{\eta}_1$' % numpy.log10(optimal_eta),
                     horizontalalignment='left', verticalalignment='bottom',
                     fontsize=10)

        for i in range(len(asym_maxima_ord2)):
            ax2.text(asym_maxima_ord2[i]*10**0.05, min_plot_lim*0.05,
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
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    lg = plt.legend(handles, labels, frameon=False, fontsize='small',
                    loc='upper left', bbox_to_anchor=(1.2, 1.04))

    # Save plots
    plt.tight_layout()
    filename = 'likelihood_first_derivative'
    save_plot(plt, filename, transparent_background=False, pdf=True,
              bbox_extra_artists=(lg, ))

    plt.show()
