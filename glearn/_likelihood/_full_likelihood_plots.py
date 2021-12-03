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

def plot(full_likelihood, result):
    """
    Plot likelihood function and its derivatives.
    """

    # Plot likelihood for scale, fixed sigma and sigma0
    _plot_likelihood_versus_scale(
            full_likelihood, result, other_sigmas=numpy.logspace(-1, 1, 3))

    # Plot likelihood for sigma, fixed sigma0 and scale
    _plot_likelihood_versus_sigma(
            full_likelihood, result, other_scales=numpy.logspace(-1, 1, 3))

    # Plot likelihood for sigma0, fixed sigma and scale
    _plot_likelihood_versus_sigma0(
            full_likelihood, result, other_scales=numpy.logspace(-1, 1, 3))

    # 2d plot of likelihood versus sigma0 and sigma
    _plot_likelihood_versus_sigma0_sigma(full_likelihood, result)


# ============================
# plot likelihood versus scale
# ============================

def _plot_likelihood_versus_scale(
        full_likelihood,
        result,
        other_sigmas=None):
    """
    Plots log likelihood versus scale hyperparameter. Other hyperparameters
    such as sigma and sigma0 are fixed. sigma is used by both its optimal value
    and user-defined values and plots are iterated by the multiple sigma
    values. On the other hand, sigma0 is only used from its optimal value.
    """

    # This function can only plot one dimensional data.
    dimension = full_likelihood.cov.mixed_cor.cor.dimension
    if dimension != 1:
        raise ValueError('To plot likelihood w.r.t "eta" and "scale", ' +
                         'the dimension of the data points should be one.')

    load_plot_settings()

    # Optimal point
    optimal_sigma = result['hyperparam']['sigma']
    optimal_sigma0 = result['hyperparam']['sigma0']

    # Convert sigma to a numpy array
    if other_sigmas is not None:
        if numpy.isscalar(other_sigmas):
            other_sigmas = numpy.array([other_sigmas])
        elif isinstance(other_sigmas, list):
            other_sigmas = numpy.array(other_sigmas)
        elif not isinstance(other_sigmas, numpy.ndarray):
            raise TypeError('"other_sigmas" should be either a scalar, list,' +
                            'or numpy.ndarray.')

    # Concatenate all given sigmas
    if other_sigmas is not None:
        sigmas = numpy.r_[optimal_sigma, other_sigmas]
    else:
        sigmas = numpy.r_[optimal_sigma]
    sigmas = numpy.sort(sigmas)

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
    if full_likelihood.use_log_scale:
        scale_x = numpy.log10(scale)
    else:
        scale_x = scale

    d0_ell_perturb_sigma = numpy.zeros((stencil_size, sigmas.size, scale.size),
                                       dtype=float)
    d0_ell_perturb_sigma0 = numpy.zeros((stencil_size, sigmas.size,
                                        scale.size), dtype=float)
    d1_ell = numpy.zeros((sigmas.size, scale.size), dtype=float)
    d2_ell = numpy.zeros((sigmas.size, scale.size), dtype=float)
    d2_mixed_sigma_ell = numpy.zeros((sigmas.size, scale.size), dtype=float)
    d2_mixed_sigma0_ell = numpy.zeros((sigmas.size, scale.size), dtype=float)
    d1_ell_perturb_sigma_numerical = numpy.zeros(
            (stencil_size, sigmas.size, scale.size-2), dtype=float)
    d1_ell_perturb_sigma0_numerical = numpy.zeros(
            (stencil_size, sigmas.size, scale.size-2), dtype=float)
    d2_ell_numerical = numpy.zeros((sigmas.size, scale.size-4), dtype=float)
    d2_mixed_sigma_ell_numerical = numpy.zeros((sigmas.size, scale.size-2),
                                               dtype=float)
    d2_mixed_sigma0_ell_numerical = numpy.zeros((sigmas.size, scale.size-2),
                                                dtype=float)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    colors = matplotlib.cm.nipy_spectral(numpy.linspace(0, 0.9, sigmas.size))

    for i in range(sigmas.size):

        if full_likelihood.use_log_sigmas:
            # Stencil to perturb sigma
            log_sigma = numpy.log10(sigmas[i])
            d_sigma = numpy.max([numpy.abs(log_sigma) * 1e-3, 1e-4])
            sigma_stencil = 10.0**(
                log_sigma + d_sigma *
                numpy.arange(-stencil_size//2+1, stencil_size//2+1))

            # Stencil to perturb sigma0
            log_sigma0 = numpy.log10(optimal_sigma0)
            d_sigma0 = numpy.max([numpy.abs(log_sigma0) * 1e-3, 1e-4])
            sigma0_stencil = 10.0**(
                log_sigma0 + d_sigma0 *
                numpy.arange(-stencil_size//2+1, stencil_size//2+1))

        else:
            # Stencil to perturb sigma
            d_sigma = sigmas[i] * 1e-3
            sigma_stencil = sigmas[i] + d_sigma * \
                numpy.arange(-stencil_size//2+1, stencil_size//2+1)

            # Stencil to perturb sigma0
            d_sigma0 = optimal_sigma0 * 1e-3
            sigma0_stencil = optimal_sigma0 + d_sigma0 * \
                numpy.arange(-stencil_size//2+1, stencil_size//2+1)

        for j in range(scale.size):

            # Set the scale
            full_likelihood.cov.set_scale(scale[j])

            # Likelihood (and its perturbation w.r.t sigma)
            for k in range(stencil_size):
                hyperparam = numpy.r_[
                        sigma_stencil[k], optimal_sigma0, scale[j]]
                hyperparam = full_likelihood.hyperparam_to_log_hyperparam(
                            hyperparam)
                sign_switch = False
                d0_ell_perturb_sigma[k, i, j] = full_likelihood.likelihood(
                        sign_switch, hyperparam)

            # Likelihood (and its perturbation w.r.t sigma0)
            for k in range(stencil_size):
                hyperparam = numpy.r_[
                        sigmas[i], sigma0_stencil[k], scale[j]]
                hyperparam = full_likelihood.hyperparam_to_log_hyperparam(
                            hyperparam)
                sign_switch = False
                d0_ell_perturb_sigma0[k, i, j] = full_likelihood.likelihood(
                        sign_switch, hyperparam)

            # First derivative of likelihood w.r.t distance scale
            hyperparam = numpy.r_[
                    sigmas[i], optimal_sigma0, scale[j]]
            hyperparam = full_likelihood.hyperparam_to_log_hyperparam(
                    hyperparam)
            jacobian_ = full_likelihood.likelihood_jacobian(sign_switch,
                                                            hyperparam)
            d1_ell[i, j] = jacobian_[2]

            # Second derivative of likelihood w.r.t distance scale
            hessian_ = full_likelihood.likelihood_hessian(sign_switch,
                                                          hyperparam)
            d2_mixed_sigma_ell[i, j] = hessian_[0, 2]
            d2_mixed_sigma0_ell[i, j] = hessian_[1, 2]
            d2_ell[i, j] = hessian_[2, 2]

        for k in range(stencil_size):
            # Compute first derivative numerically (perturb sigma)
            d1_ell_perturb_sigma_numerical[k, i, :] = \
                (d0_ell_perturb_sigma[k, i, 2:] -
                    d0_ell_perturb_sigma[k, i, :-2]) / \
                (scale_x[2:] - scale_x[:-2])

            # Compute first derivative numerically (perturb sigma0)
            d1_ell_perturb_sigma0_numerical[k, i, :] = \
                (d0_ell_perturb_sigma0[k, i, 2:] -
                    d0_ell_perturb_sigma0[k, i, :-2]) / \
                (scale_x[2:] - scale_x[:-2])

            # Compute second mixed derivative w.r.t sigma, numerically
            d2_mixed_sigma_ell_numerical[i, :] += \
                coeff[k] * d1_ell_perturb_sigma_numerical[k, i, :] / d_sigma

            # Compute second mixed derivative w.r.t sigma0, numerically
            d2_mixed_sigma0_ell_numerical[i, :] += \
                coeff[k] * d1_ell_perturb_sigma0_numerical[k, i, :] / d_sigma0

        # Note, the above mixed derivatives are w.r.t sigma and sigma0. To
        # compute the derivatives w.r.t to sigma**2 and sigma0**2 (squared
        # variables) divide them by 2*sigma and 2*sigma0 respectively.
        if full_likelihood.use_log_sigmas:
            d2_mixed_sigma_ell_numerical[i, :] /= 2.0
            d2_mixed_sigma0_ell_numerical[i, :] /= 2.0
        else:
            d2_mixed_sigma_ell_numerical[i, :] /= (2.0 * sigmas[i])
            d2_mixed_sigma0_ell_numerical[i, :] /= (2.0 * optimal_sigma0)

        # Compute second derivative numerically
        d2_ell_numerical[i, :] = \
            (d1_ell_perturb_sigma_numerical[center_stencil, i, 2:] -
             d1_ell_perturb_sigma_numerical[center_stencil, i, :-2]) / \
            (scale_x[3:-1] - scale_x[1:-3])

        # Find maximum of ell
        max_index = numpy.argmax(d0_ell_perturb_sigma[center_stencil, i, :])
        optimal_scale = scale[max_index]
        optimal_ell = d0_ell_perturb_sigma[center_stencil, i, max_index]

        # Plot
        if sigmas[i] == optimal_sigma:
            label = r'$\hat{\sigma}=%0.2e$' % sigmas[i]
            marker = 'X'
        else:
            label = r'$\sigma=%0.2e$' % sigmas[i]
            marker = 'o'
        ax[0, 0].plot(scale, d0_ell_perturb_sigma[center_stencil, i, :],
                      color=colors[i], label=label)
        ax[0, 1].plot(scale, d1_ell[i, :], color=colors[i], label=label)
        ax[0, 2].plot(scale, d2_ell[i, :], color=colors[i], label=label)
        ax[1, 0].plot(scale, d2_mixed_sigma_ell[i, :], color=colors[i],
                      label=label)
        ax[1, 1].plot(scale, d2_mixed_sigma0_ell[i, :], color=colors[i],
                      label=label)
        ax[0, 1].plot(scale[1:-1],
                      d1_ell_perturb_sigma_numerical[center_stencil, i, :],
                      '--', color=colors[i])
        ax[0, 2].plot(scale[2:-2], d2_ell_numerical[i, :], '--',
                      color=colors[i])
        ax[1, 0].plot(scale[1:-1], d2_mixed_sigma_ell_numerical[i, :],
                      '--', color=colors[i])
        ax[1, 1].plot(scale[1:-1], d2_mixed_sigma0_ell_numerical[i, :],
                      '--', color=colors[i])
        p = ax[0, 0].plot(optimal_scale, optimal_ell, marker, color=colors[i],
                          markersize=3)
        ax[0, 1].plot(optimal_scale, 0.0,  marker, color=colors[i],
                      markersize=3)

    ax[0, 0].legend(p, [r'optimal $\theta$'])
    ax[0, 0].legend(loc='lower right')
    ax[0, 1].legend(loc='lower right')
    ax[0, 2].legend(loc='lower right')
    ax[1, 0].legend(loc='lower right')
    ax[1, 1].legend(loc='lower right')
    ax[0, 0].set_xscale('log')
    ax[0, 1].set_xscale('log')
    ax[0, 2].set_xscale('log')
    ax[1, 0].set_xscale('log')
    ax[1, 1].set_xscale('log')
    ax[0, 0].set_yscale('linear')
    ax[0, 1].set_yscale('linear')
    ax[0, 2].set_yscale('linear')
    ax[1, 0].set_yscale('linear')
    ax[1, 1].set_yscale('linear')

    # Plot annotations
    ax[0, 0].set_xlim([scale[0], scale[-1]])
    ax[0, 1].set_xlim([scale[0], scale[-1]])
    ax[0, 2].set_xlim([scale[0], scale[-1]])
    ax[1, 0].set_xlim([scale[0], scale[-1]])
    ax[1, 1].set_xlim([scale[0], scale[-1]])
    ax[0, 0].set_xlabel(r'$\theta$')
    ax[0, 1].set_xlabel(r'$\theta$')
    ax[0, 2].set_xlabel(r'$\theta$')
    ax[1, 0].set_xlabel(r'$\theta$')
    ax[1, 1].set_xlabel(r'$\theta$')
    ax[0, 0].set_ylabel(r'$\ell(\theta | \sigma^2, \sigma_0^2)$')

    if full_likelihood.use_log_scale:
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d} \ell(\theta | \sigma^2, \sigma_0^2)} ' +
            r'{\mathrm{d} (\ln \theta)}$')
    else:
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d} \ell(\theta | \sigma^2, \sigma_0^2)} ' +
            r'{\mathrm{d} \theta}$')

    if full_likelihood.use_log_scale:
        ax[0, 2].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
            r'{\mathrm{d} (\ln \theta)^2}$')
    else:
        ax[0, 2].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
            r'{\mathrm{d} \theta^2}$')

    if full_likelihood.use_log_scale:
        if full_likelihood.use_log_sigmas:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} (\ln \theta) \mathrm{d} (\ln \sigma^2)}$')
        else:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} (\ln \theta) \mathrm{d} \sigma^2}$')
    else:
        if full_likelihood.use_log_sigmas:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} \theta \mathrm{d} (\ln \sigma^2)}$')
        else:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} \theta \mathrm{d} \sigma^2}$')

    if full_likelihood.use_log_scale:
        if full_likelihood.use_log_sigmas:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} (\ln \theta) \mathrm{d} (\ln {\sigma_0}^2)}$')
        else:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} (\ln \theta) \mathrm{d} {\sigma_0}^2}$')
    else:
        if full_likelihood.use_log_sigmas:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} \theta \mathrm{d} (\ln {\sigma_0}^2)}$')
        else:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} \theta \mathrm{d} {\sigma_0}^2}$')

    ax[0, 0].set_title(r'Log likelihood function, given ' +
                       r'$(\sigma^2, \sigma_0^2)$ ')
    ax[0, 1].set_title(r'First derivative of log likelihood, given ' +
                       r'$(\sigma^2, \sigma_0^2)$')
    ax[0, 2].set_title(r'Second derivative of log likelihood, given' +
                       r'$(\sigma^2, \sigma_0^2)$')
    ax[1, 0].set_title(r'Second mixed derivative of log likelihood, ' +
                       r'given $(\sigma^2, \sigma_0^2)$')
    ax[1, 1].set_title(r'Second mixed derivative of log likelihood, ' +
                       r'given $(\sigma^2, \sigma_0^2)$')
    ax[0, 0].grid(True, which='both')
    ax[0, 1].grid(True, which='both')
    ax[0, 2].grid(True, which='both')
    ax[1, 0].grid(True, which='both')
    ax[1, 1].grid(True, which='both')

    ax[1, 2].set_axis_off()

    plt.tight_layout()
    plt.show()


# ============================
# plot likelihood versus sigma
# ============================

def _plot_likelihood_versus_sigma(
        full_likelihood,
        result,
        other_scales=None):
    """
    Plots log likelihood versus sigma. Other hyperparameters are fixed. Also,
    scale is used from both its optimal value and user-defined values. Plots
    are iterated over multiple values of scale. On the other hand, sigma0 is
    fixed to its optimal value.
    """

    # This function can only plot one dimensional data.
    dimension = full_likelihood.cov.mixed_cor.cor.dimension
    if dimension != 1:
        raise ValueError('To plot likelihood w.r.t "eta" and "scale", ' +
                         'the dimension of the data points should be one.')

    load_plot_settings()

    # Optimal point
    optimal_sigma = result['hyperparam']['sigma']
    optimal_sigma0 = result['hyperparam']['sigma0']
    optimal_scale = result['hyperparam']['scale']

    # Convert scale to a numpy array
    if other_scales is not None:
        if numpy.isscalar(other_scales):
            other_scales = numpy.array([other_scales])
        elif isinstance(other_scales, list):
            other_scales = numpy.array(other_scales)
        elif not isinstance(other_scales, numpy.ndarray):
            raise TypeError('"other_scales" should be either a scalar, list,' +
                            'or numpy.ndarray.')

    # Concatenate all given scales
    if other_scales is not None:
        scales = numpy.r_[optimal_scale, other_scales]
    else:
        scales = numpy.r_[optimal_scale]
    scales = numpy.sort(scales)

    # 2nd or 4th order finite difference coefficients for first derivative
    coeff = numpy.array([-1.0/2.0, 0.0, 1.0/2.0])
    # coeff = numpy.array([1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0])

    # The first axis of some arrays below is 3, used for varying eta for
    # numerical evaluating of mixed derivative w.r.t eta.
    stencil_size = coeff.size
    center_stencil = stencil_size//2  # Index of the center of stencil

    # Generate ell for various sigma
    sigma = numpy.logspace(-2, 2, 100)

    # The variable on the abscissa to take derivative with respect to it.
    if full_likelihood.use_log_sigmas:
        x_sigma = numpy.log10(sigma)
    else:
        x_sigma = sigma

    d0_ell_perturb_scale = numpy.zeros((stencil_size, scales.size, sigma.size),
                                       dtype=float)
    d0_ell_perturb_sigma0 = numpy.zeros(
            (stencil_size, scales.size, sigma.size), dtype=float)
    d1_ell = numpy.zeros((scales.size, sigma.size), dtype=float)
    d2_ell = numpy.zeros((scales.size, sigma.size), dtype=float)
    d2_mixed_scale_ell = numpy.zeros((scales.size, sigma.size), dtype=float)
    d2_mixed_sigma0_ell = numpy.zeros((scales.size, sigma.size), dtype=float)
    d1_ell_perturb_scale_numerical = numpy.zeros(
            (stencil_size, scales.size, sigma.size-2), dtype=float)
    d1_ell_perturb_sigma0_numerical = numpy.zeros(
            (stencil_size, scales.size, sigma.size-2), dtype=float)
    d2_ell_numerical = numpy.zeros((scales.size, sigma.size-4), dtype=float)
    d2_mixed_scale_ell_numerical = numpy.zeros((scales.size, sigma.size-2),
                                               dtype=float)
    d2_mixed_sigma0_ell_numerical = numpy.zeros((scales.size, sigma.size-2),
                                                dtype=float)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    colors = matplotlib.cm.nipy_spectral(numpy.linspace(0, 0.9, scales.size))

    for i in range(scales.size):

        # Stencil to perturb scale
        if full_likelihood.use_log_scale:
            log_scale = numpy.log10(scales[i])
            d_scale = numpy.max([numpy.abs(log_scale) * 1e-3, 1e-4])
            scale_stencil = 10.0**(
                log_scale + d_scale *
                numpy.arange(-stencil_size//2+1, stencil_size//2+1))
        else:
            d_scale = scales[i] * 1e-3
            scale_stencil = scales[i] + d_scale * \
                numpy.arange(-stencil_size//2+1, stencil_size//2+1)

        # Stencil to perturb sigma0
        if full_likelihood.use_log_sigmas:
            log_sigma0 = numpy.log10(optimal_sigma0)
            d_sigma0 = numpy.max([numpy.abs(log_sigma0) * 1e-3, 1e-4])
            sigma0_stencil = 10.0**(
                    log_sigma0 + d_sigma0 *
                    numpy.arange(-stencil_size//2+1, stencil_size//2+1))
        else:
            d_sigma0 = optimal_sigma0 * 1e-3
            sigma0_stencil = optimal_sigma0 + d_sigma0 * \
                numpy.arange(-stencil_size//2+1, stencil_size//2+1)

        # Likelihood (and its perturbation w.r.t sigma)
        for k in range(stencil_size):

            # Set the scale
            full_likelihood.cov.set_scale(scale_stencil[k])

            for j in range(sigma.size):
                hyperparam = numpy.r_[
                        sigma[j], optimal_sigma0, scale_stencil[k]]
                hyperparam = full_likelihood.hyperparam_to_log_hyperparam(
                            hyperparam)
                sign_switch = False
                d0_ell_perturb_scale[k, i, j] = full_likelihood.likelihood(
                    sign_switch, hyperparam)

        # Likelihood (and its perturbation w.r.t sigma0)
        full_likelihood.cov.set_scale(scales[i])
        for k in range(stencil_size):
            for j in range(sigma.size):
                hyperparam = numpy.r_[sigma[j], sigma0_stencil[k], scales[i]]
                hyperparam = full_likelihood.hyperparam_to_log_hyperparam(
                            hyperparam)
                sign_switch = False
                d0_ell_perturb_sigma0[k, i, j] = full_likelihood.likelihood(
                        sign_switch, hyperparam)

        # First derivative of likelihood w.r.t distance scale
        for j in range(sigma.size):
            hyperparam = numpy.r_[sigma[j], optimal_sigma0, scales[i]]
            hyperparam = full_likelihood.hyperparam_to_log_hyperparam(
                        hyperparam)
            jacobian_ = full_likelihood.likelihood_jacobian(sign_switch,
                                                            hyperparam)
            d1_ell[i, j] = jacobian_[0]

            # Second derivative of likelihood w.r.t distance scale
            hessian_ = full_likelihood.likelihood_hessian(sign_switch,
                                                          hyperparam)
            d2_mixed_scale_ell[i, j] = hessian_[0, 2]
            d2_mixed_sigma0_ell[i, j] = hessian_[0, 1]
            d2_ell[i, j] = hessian_[0, 0]

        for k in range(stencil_size):
            # First derivative numerically (perturb scale)
            d1_ell_perturb_scale_numerical[k, i, :] = \
                (d0_ell_perturb_scale[k, i, 2:] -
                    d0_ell_perturb_scale[k, i, :-2]) / \
                (x_sigma[2:] - x_sigma[:-2])

            # To take derivative w.r.t sigma**2, divide by 2*sigma.
            for j in range(sigma.size-2):
                if full_likelihood.use_log_sigmas:
                    d1_ell_perturb_scale_numerical[k, i, j] /= 2.0
                else:
                    d1_ell_perturb_scale_numerical[k, i, j] /= \
                            (2.0 * sigma[j+1])

            # Compute first derivative numerically (perturb sigma0)
            d1_ell_perturb_sigma0_numerical[k, i, :] = \
                (d0_ell_perturb_sigma0[k, i, 2:] -
                    d0_ell_perturb_sigma0[k, i, :-2]) / \
                (x_sigma[2:] - x_sigma[:-2])

            # To take derivative w.r.t sigma**2, divide by 2*sigma.
            for j in range(sigma.size-2):
                if full_likelihood.use_log_sigmas:
                    d1_ell_perturb_sigma0_numerical[k, i, j] /= 2.0
                else:
                    d1_ell_perturb_sigma0_numerical[k, i, j] /= \
                            (2.0 * sigma[j+1])

            # Second mixed derivative w.r.t scale, numerically
            d2_mixed_scale_ell_numerical[i, :] += coeff[k] * \
                d1_ell_perturb_scale_numerical[k, i, :] / d_scale

            # Compute second mixed derivative w.r.t sigma0, numerically
            d2_mixed_sigma0_ell_numerical[i, :] += \
                coeff[k] * d1_ell_perturb_sigma0_numerical[k, i, :] / d_sigma0

        # To take derivative w.r.t sigma0**2, divide by 2*sigma0.
        if full_likelihood.use_log_sigmas:
            d2_mixed_sigma0_ell_numerical[i, :] /= 2.0
        else:
            d2_mixed_sigma0_ell_numerical[i, :] /= (2.0 * optimal_sigma0)

        # Compute second derivative numerically
        d2_ell_numerical[i, :] = \
            (d1_ell_perturb_scale_numerical[center_stencil, i, 2:] -
             d1_ell_perturb_scale_numerical[center_stencil, i, :-2]) / \
            (x_sigma[3:-1] - x_sigma[1:-3])

        # To take derivative w.r.t sigma0**2, divide by 2*sigma0.
        for j in range(sigma.size-4):
            if full_likelihood.use_log_sigmas:
                d2_ell_numerical[i, j] /= 2.0
            else:
                d2_ell_numerical[i, j] /= (2.0 * sigma[j+2])

        # Find maximum of ell
        max_index = numpy.argmax(d0_ell_perturb_scale[center_stencil, i, :])
        optimal_sigma = sigma[max_index]
        optimal_ell = d0_ell_perturb_scale[center_stencil, i, max_index]

        # Plot
        if any(scales[i] == optimal_scale):
            label = r'$\hat{\theta}=%0.2e$' % scales[i]
            marker = 'X'
        else:
            label = r'$\theta=%0.2e$' % scales[i]
            marker = 'o'
        ax[0, 0].plot(sigma, d0_ell_perturb_scale[center_stencil, i, :],
                      color=colors[i], label=label)
        ax[0, 1].plot(sigma, d1_ell[i, :], color=colors[i], label=label)
        ax[0, 2].plot(sigma, d2_ell[i, :], color=colors[i], label=label)
        ax[1, 0].plot(sigma, d2_mixed_scale_ell[i, :], color=colors[i],
                      label=label)
        ax[1, 1].plot(sigma, d2_mixed_sigma0_ell[i, :], color=colors[i],
                      label=label)
        ax[0, 1].plot(sigma[1:-1], d1_ell_perturb_scale_numerical[
                      center_stencil, i, :], '--', color=colors[i])
        ax[0, 2].plot(sigma[2:-2], d2_ell_numerical[i, :], '--',
                      color=colors[i])
        ax[1, 0].plot(sigma[1:-1], d2_mixed_scale_ell_numerical[i, :],
                      '--', color=colors[i])
        ax[1, 1].plot(sigma[1:-1], d2_mixed_sigma0_ell_numerical[i, :],
                      '--', color=colors[i])
        p = ax[0, 0].plot(optimal_sigma, optimal_ell, marker,
                          color=colors[i], markersize=3)
        ax[0, 1].plot(optimal_sigma, 0.0,  marker, color=colors[i],
                      markersize=3)

    ax[0, 0].legend(p, [r'optimal $\sigma$'])
    ax[0, 0].legend(loc='lower right')
    ax[0, 1].legend(loc='lower right')
    ax[0, 2].legend(loc='lower right')
    ax[1, 0].legend(loc='lower right')
    ax[1, 1].legend(loc='lower right')
    ax[0, 0].set_xscale('log')
    ax[0, 1].set_xscale('log')
    ax[0, 2].set_xscale('log')
    ax[1, 0].set_xscale('log')
    ax[1, 1].set_xscale('log')
    ax[0, 0].set_yscale('linear')
    ax[0, 1].set_yscale('linear')
    ax[0, 2].set_yscale('linear')
    ax[1, 0].set_yscale('linear')
    ax[1, 1].set_yscale('linear')

    # Plot annotations
    ax[0, 0].set_xlim([sigma[0], sigma[-1]])
    ax[0, 1].set_xlim([sigma[0], sigma[-1]])
    ax[0, 2].set_xlim([sigma[0], sigma[-1]])
    ax[1, 0].set_xlim([sigma[0], sigma[-1]])
    ax[1, 1].set_xlim([sigma[0], sigma[-1]])
    ax[0, 0].set_xlabel(r'$\sigma$')
    ax[0, 1].set_xlabel(r'$\sigma$')
    ax[0, 2].set_xlabel(r'$\sigma$')
    ax[1, 0].set_xlabel(r'$\sigma$')
    ax[1, 1].set_xlabel(r'$\sigma$')
    ax[0, 0].set_ylabel(r'$\ell(\sigma^2 | \sigma_0^2, \theta)$')

    if full_likelihood.use_log_sigmas:
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d} \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
            r'{\mathrm{d} (\ln \sigma^2)}$')
    else:
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d} \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
            r'{\mathrm{d} \sigma^2}$')

    if full_likelihood.use_log_sigmas:
        ax[0, 2].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
            r'{\mathrm{d} (\ln \sigma^2)^2}$')
    else:
        ax[0, 2].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
            r'{\mathrm{d} (\sigma^2)^2}$')

    if full_likelihood.use_log_scale:
        if full_likelihood.use_log_sigmas:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
                r'{\mathrm{d} (\ln \sigma^2) \mathrm{d} (\ln \theta)}$')
        else:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
                r'{\mathrm{d} \sigma^2 \mathrm{d} (\ln \theta)}$')
    else:
        if full_likelihood.use_log_sigmas:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
                r'{\mathrm{d} (\ln \sigma^2) \mathrm{d} \theta}$')
        else:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
                r'{\mathrm{d} \sigma^2 \mathrm{d} \theta}$')

    if full_likelihood.use_log_sigmas:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
            r'{\mathrm{d} (\ln \sigma^2) \mathrm{d} (\ln {\sigma_0}^2)}$')
    else:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
            r'{\mathrm{d} \sigma^2 \mathrm{d} {\sigma_0}^2}$')

    ax[0, 0].set_title(r'Log likelihood function, given ' +
                       r'$(\sigma_0^2, \theta)$')
    ax[0, 1].set_title(r'First derivative of log likelihood, given ' +
                       r'$(\sigma_0^2, \theta)$')
    ax[0, 2].set_title(r'Second derivative of log likelihood, given ' +
                       r'$(\sigma_0^2, \theta)$')
    ax[1, 0].set_title(r'Second mixed derivative of log likelihood, given ' +
                       r'$(\sigma_0^2, \theta)$')
    ax[1, 1].set_title(r'Second mixed derivative of log likelihood, given ' +
                       r'$(\sigma_0^2, \theta)$')
    ax[0, 0].grid(True, which='both')
    ax[0, 1].grid(True, which='both')
    ax[0, 2].grid(True, which='both')
    ax[1, 0].grid(True, which='both')
    ax[1, 1].grid(True, which='both')

    ax[1, 2].set_axis_off()

    plt.tight_layout()
    plt.show()


# =============================
# plot likelihood versus sigma0
# =============================

def _plot_likelihood_versus_sigma0(
        full_likelihood,
        result,
        other_scales=None):
    """
    Plots log likelihood versus sigma0. Other hyperparameters are fixed. Also,
    scale is used from both its optimal value and user-defined values. Plots
    are iterated over multiple values of scale. On the other hand, sigma is
    fixed to its optimal value.
    """

    # This function can only plot one dimensional data.
    dimension = full_likelihood.cov.mixed_cor.cor.dimension
    if dimension != 1:
        raise ValueError('To plot likelihood w.r.t "eta" and "scale", ' +
                         'the dimension of the data points should be one.')

    load_plot_settings()

    # Optimal point
    optimal_sigma = result['hyperparam']['sigma']
    optimal_sigma0 = result['hyperparam']['sigma0']
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

    # Concatenate all given scales
    if other_scales is not None:
        scales = numpy.r_[optimal_scale, other_scales]
    else:
        scales = numpy.r_[optimal_scale]
    scales = numpy.sort(scales)

    # 2nd or 4th order finite difference coefficients for first derivative
    coeff = numpy.array([-1.0/2.0, 0.0, 1.0/2.0])
    # coeff = numpy.array([1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0])

    # The first axis of some arrays below is 3, used for varying eta for
    # numerical evaluating of mixed derivative w.r.t eta.
    stencil_size = coeff.size
    center_stencil = stencil_size//2  # Index of the center of stencil

    # Generate ell for various sigma0
    sigma0 = numpy.logspace(-2, 2, 100)

    # The variable on the abscissa to take derivative with respect to it.
    if full_likelihood.use_log_sigmas:
        x_sigma0 = numpy.log10(sigma0)
    else:
        x_sigma0 = sigma0

    d0_ell_perturb_scale = numpy.zeros(
            (stencil_size, scales.size, sigma0.size), dtype=float)
    d0_ell_perturb_sigma = numpy.zeros(
            (stencil_size, scales.size, sigma0.size), dtype=float)
    d1_ell = numpy.zeros((scales.size, sigma0.size), dtype=float)
    d2_ell = numpy.zeros((scales.size, sigma0.size), dtype=float)
    d2_mixed_scale_ell = numpy.zeros((scales.size, sigma0.size), dtype=float)
    d2_mixed_sigma_ell = numpy.zeros((scales.size, sigma0.size), dtype=float)
    d1_ell_perturb_scale_numerical = numpy.zeros(
            (stencil_size, scales.size, sigma0.size-2), dtype=float)
    d1_ell_perturb_sigma_numerical = numpy.zeros(
            (stencil_size, scales.size, sigma0.size-2), dtype=float)
    d2_ell_numerical = numpy.zeros((scales.size, sigma0.size-4), dtype=float)
    d2_mixed_scale_ell_numerical = numpy.zeros(
            (scales.size, sigma0.size-2), dtype=float)
    d2_mixed_sigma_ell_numerical = numpy.zeros(
            (scales.size, sigma0.size-2), dtype=float)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    colors = matplotlib.cm.nipy_spectral(numpy.linspace(0, 0.9, scales.size))

    for i in range(scales.size):

        # Stencil to perturb scale
        if full_likelihood.use_log_scale:
            log_scale = numpy.log10(scales[i])
            d_scale = numpy.max([numpy.abs(log_scale) * 1e-3, 1e-4])
            scale_stencil = 10.0**(
                log_scale + d_scale *
                numpy.arange(-stencil_size//2+1, stencil_size//2+1))
        else:
            d_scale = scales[i] * 1e-3
            scale_stencil = scales[i] + d_scale * \
                numpy.arange(-stencil_size//2+1, stencil_size//2+1)

        # Stencil to perturb sigma
        if full_likelihood.use_log_sigmas:
            log_sigma = numpy.log10(optimal_sigma)
            d_sigma = numpy.max([numpy.abs(log_sigma) * 1e-3, 1e-4])
            sigma_stencil = 10.0**(
                    log_sigma + d_sigma *
                    numpy.arange(-stencil_size//2+1, stencil_size//2+1))
        else:
            d_sigma = optimal_sigma * 1e-3
            sigma_stencil = optimal_sigma + d_sigma * \
                numpy.arange(-stencil_size//2+1, stencil_size//2+1)

        # Likelihood (and its perturbation w.r.t sigma0)
        for k in range(stencil_size):

            # Set the scale
            full_likelihood.cov.set_scale(scale_stencil[k])

            for j in range(sigma0.size):
                hyperparam = numpy.r_[
                        optimal_sigma, sigma0[j], scale_stencil[k]]
                hyperparam = full_likelihood.hyperparam_to_log_hyperparam(
                            hyperparam)
                sign_switch = False
                d0_ell_perturb_scale[k, i, j] = full_likelihood.likelihood(
                    sign_switch, hyperparam)

        # Likelihood (and its perturbation w.r.t sigma)
        full_likelihood.cov.set_scale(scales[i])
        for k in range(stencil_size):
            for j in range(sigma0.size):
                hyperparam = numpy.r_[sigma_stencil[k], sigma0[j], scales[i]]
                hyperparam = full_likelihood.hyperparam_to_log_hyperparam(
                            hyperparam)
                sign_switch = False
                d0_ell_perturb_sigma[k, i, j] = full_likelihood.likelihood(
                    sign_switch, hyperparam)

        # First derivative of likelihood w.r.t distance scale
        for j in range(sigma0.size):
            hyperparam = numpy.r_[sigma_stencil[k], sigma0[j], scales[i]]
            hyperparam = full_likelihood.hyperparam_to_log_hyperparam(
                        hyperparam)
            jacobian_ = full_likelihood.likelihood_jacobian(sign_switch,
                                                            hyperparam)
            d1_ell[i, j] = jacobian_[1]

            # Second derivative of likelihood w.r.t distance scale
            hessian_ = full_likelihood.likelihood_hessian(sign_switch,
                                                          hyperparam)
            d2_mixed_scale_ell[i, j] = hessian_[1, 2]
            d2_mixed_sigma_ell[i, j] = hessian_[1, 0]
            d2_ell[i, j] = hessian_[1, 1]

        for k in range(stencil_size):
            # First derivative numerically (perturb scale)
            d1_ell_perturb_scale_numerical[k, i, :] = \
                (d0_ell_perturb_scale[k, i, 2:] -
                    d0_ell_perturb_scale[k, i, :-2]) / \
                (x_sigma0[2:] - x_sigma0[:-2])

            # To take derivative w.r.t sigma**2, divide by 2*sigma0.
            for j in range(sigma0.size-2):
                if full_likelihood.use_log_sigmas:
                    d1_ell_perturb_scale_numerical[k, i, j] /= 2.0
                else:
                    d1_ell_perturb_scale_numerical[k, i, j] /= \
                            (2.0 * sigma0[j+1])

            # Compute first derivative numerically (perturb sigma)
            d1_ell_perturb_sigma_numerical[k, i, :] = \
                (d0_ell_perturb_sigma[k, i, 2:] -
                    d0_ell_perturb_sigma[k, i, :-2]) / \
                (x_sigma0[2:] - x_sigma0[:-2])

            # To take derivative w.r.t sigma0**2, divide by 2*sigma0.
            for j in range(sigma0.size-2):
                if full_likelihood.use_log_sigmas:
                    d1_ell_perturb_sigma_numerical[k, i, j] /= 2.0
                else:
                    d1_ell_perturb_sigma_numerical[k, i, j] /= \
                            (2.0 * sigma0[j+1])

            # Second mixed derivative w.r.t scale, numerically
            d2_mixed_scale_ell_numerical[i, :] += coeff[k] * \
                d1_ell_perturb_scale_numerical[k, i, :] / d_scale

            # Compute second mixed derivative w.r.t sigma, numerically
            d2_mixed_sigma_ell_numerical[i, :] += \
                coeff[k] * d1_ell_perturb_sigma_numerical[k, i, :] / d_sigma

        # To take derivative w.r.t sigma**2, divide by 2*sigma.
        if full_likelihood.use_log_sigmas:
            d2_mixed_sigma_ell_numerical[i, :] /= 2.0
        else:
            d2_mixed_sigma_ell_numerical[i, :] /= (2.0 * optimal_sigma)

        # Compute second derivative numerically
        d2_ell_numerical[i, :] = \
            (d1_ell_perturb_scale_numerical[center_stencil, i, 2:] -
             d1_ell_perturb_scale_numerical[center_stencil, i, :-2]) / \
            (x_sigma0[3:-1] - x_sigma0[1:-3])

        # To take derivative w.r.t sigma**2, divide by 2*sigma0.
        for j in range(sigma0.size-4):
            if full_likelihood.use_log_sigmas:
                d2_ell_numerical[i, j] /= 2.0
            else:
                d2_ell_numerical[i, j] /= (2.0 * sigma0[j+2])

        # Find maximum of ell
        max_index = numpy.argmax(
                d0_ell_perturb_scale[center_stencil, i, :])
        optimal_sigma0 = sigma0[max_index]
        optimal_ell = d0_ell_perturb_scale[center_stencil, i, max_index]

        # Plot
        if any(scales[i] == optimal_scale):
            label = r'$\hat{\theta}=%0.2e$' % scales[i]
            marker = 'X'
        else:
            label = r'$\theta=%0.2e$' % scales[i]
            marker = 'o'
        ax[0, 0].plot(sigma0, d0_ell_perturb_scale[center_stencil, i, :],
                      color=colors[i], label=label)
        ax[0, 1].plot(sigma0, d1_ell[i, :], color=colors[i], label=label)
        ax[0, 2].plot(sigma0, d2_ell[i, :], color=colors[i], label=label)
        ax[1, 0].plot(sigma0, d2_mixed_scale_ell[i, :], color=colors[i],
                      label=label)
        ax[1, 1].plot(sigma0, d2_mixed_sigma_ell[i, :], color=colors[i],
                      label=label)
        ax[0, 1].plot(sigma0[1:-1], d1_ell_perturb_scale_numerical[
                      center_stencil, i, :], '--', color=colors[i])
        ax[0, 2].plot(sigma0[2:-2], d2_ell_numerical[i, :], '--',
                      color=colors[i])
        ax[1, 0].plot(sigma0[1:-1], d2_mixed_scale_ell_numerical[i, :],
                      '--', color=colors[i])
        ax[1, 1].plot(sigma0[1:-1], d2_mixed_sigma_ell_numerical[i, :],
                      '--', color=colors[i])
        p = ax[0, 0].plot(optimal_sigma0, optimal_ell, marker,
                          color=colors[i], markersize=3)
        ax[0, 1].plot(optimal_sigma0, 0.0,  marker, color=colors[i],
                      markersize=3)

    ax[0, 0].legend(p, [r'optimal $\sigma0$'])
    ax[0, 0].legend(loc='lower right')
    ax[0, 1].legend(loc='lower right')
    ax[0, 2].legend(loc='lower right')
    ax[1, 0].legend(loc='lower right')
    ax[1, 1].legend(loc='lower right')
    ax[0, 0].set_xscale('log')
    ax[0, 1].set_xscale('log')
    ax[0, 2].set_xscale('log')
    ax[1, 0].set_xscale('log')
    ax[1, 1].set_xscale('log')

    # Plot annotations
    ax[0, 0].set_xlim([sigma0[0], sigma0[-1]])
    ax[0, 1].set_xlim([sigma0[0], sigma0[-1]])
    ax[0, 2].set_xlim([sigma0[0], sigma0[-1]])
    ax[1, 0].set_xlim([sigma0[0], sigma0[-1]])
    ax[1, 1].set_xlim([sigma0[0], sigma0[-1]])
    ax[0, 0].set_xlabel(r'$\sigma_0$')
    ax[0, 1].set_xlabel(r'$\sigma_0$')
    ax[0, 2].set_xlabel(r'$\sigma_0$')
    ax[1, 0].set_xlabel(r'$\sigma_0$')
    ax[1, 1].set_xlabel(r'$\sigma_0$')
    ax[0, 0].set_ylabel(r'$\ell({\sigma_0}^2 | \sigma^2, \theta)$')

    if full_likelihood.use_log_sigmas:
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d} \ell({\sigma_0}^2 | \sigma^2, \theta)} ' +
            r'{\mathrm{d} (\ln {\sigma_0}^2)}$')
    else:
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d} \ell({\sigma_0}^2 | \sigma^2, \theta)} ' +
            r'{\mathrm{d} {\sigma_0}^2}$')

    if full_likelihood.use_log_sigmas:
        ax[0, 2].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell({\sigma_0}^2 | \sigma^2, \theta)} ' +
            r'{\mathrm{d} (\ln {\sigma_0}^2)^2}$')
    else:
        ax[0, 2].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell({\sigma_0}^2 | \sigma^2, \theta)} ' +
            r'{\mathrm{d} ({\sigma_0}^2)^2}$')

    if full_likelihood.use_log_scale:
        if full_likelihood.use_log_sigmas:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell({\sigma_0}^2 | \sigma^2, ' +
                r'\theta)} {\mathrm{d} (\ln {\sigma_0}^2) \mathrm{d} ' +
                r'(\ln \theta)}$')
        else:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell({\sigma_0}^2 | \sigma^2, ' +
                r'\theta)} {\mathrm{d} {\sigma_0}^2 \mathrm{d} (\ln \theta)}$')
    else:
        if full_likelihood.use_log_sigmas:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell({\sigma_0}^2 | \sigma^2, ' +
                r'\theta)} {\mathrm{d} (\ln {\sigma_0}^2) \mathrm{d} \theta}$')
        else:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell({\sigma_0}^2 | \sigma^2, ' +
                r'\theta)} {\mathrm{d} {\sigma_0}^2 \mathrm{d} \theta}$')

    if full_likelihood.use_log_sigmas:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell({\sigma_0}^2 | \sigma^2, \theta)} ' +
            r'{\mathrm{d} (\ln {\sigma_0}^2) \mathrm{d} (\ln \sigma^2)}$')
    else:
        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell({\sigma_0}^2 | \sigma^2, \theta)} ' +
            r'{\mathrm{d} {\sigma_0}^2 \mathrm{d} \sigma^2}$')

    ax[0, 0].set_title(r'Log likelihood function, given ' +
                       r'$(\sigma^2, \theta)$')
    ax[0, 1].set_title(r'First derivative of log likelihood, given ' +
                       r'$(\sigma^2, \theta)$')
    ax[0, 2].set_title(r'Second derivative of log likelihood, given ' +
                       r'$(\sigma^2, \theta)$')
    ax[1, 0].set_title(r'Second mixed derivative of log likelihood, given ' +
                       r'$(\sigma^2, \theta)$')
    ax[1, 1].set_title(r'Second mixed derivative of log likelihood, given ' +
                       r'$(\sigma^2, \theta)$')
    ax[0, 0].grid(True, which='both')
    ax[0, 1].grid(True, which='both')
    ax[0, 2].grid(True, which='both')
    ax[1, 0].grid(True, which='both')
    ax[1, 1].grid(True, which='both')

    ax[1, 2].set_axis_off()

    plt.tight_layout()
    plt.show()


# ===================================
# plot likelihood versus sigma0 sigma
# ===================================

def _plot_likelihood_versus_sigma0_sigma(full_likelihood, result=None):
    """
    2D contour plot of log likelihood versus sigma0 and sigma.
    """

    load_plot_settings()

    # Optimal point
    optimal_sigma = result['hyperparam']['sigma']
    optimal_sigma0 = result['hyperparam']['sigma0']
    optimal_scale = result['hyperparam']['scale']
    optimal_ell = result['optimization']['max_fun']

    full_likelihood.cov.set_scale(optimal_scale)

    # Intervals cannot contain origin point as ell is minus infinity.
    sigma0 = numpy.linspace(0.02, 0.25, 50)
    sigma = numpy.linspace(0.02, 0.25, 50)
    ell = numpy.zeros((sigma0.size, sigma.size))
    for i in range(sigma0.size):
        for j in range(sigma.size):
            ell[i, j] = full_likelihood.likelihood(
                    False, numpy.array([sigma[j], sigma0[i]]))

    # Convert inf to nan
    ell = numpy.where(numpy.isinf(ell), numpy.nan, ell)

    # Smooth data for finer plot
    # sigma_ = [2, 2]  # in unit of data pixel size
    # ell = scipy.ndimage.filters.gaussian_filter(
    #         ell, sigma_, mode='nearest')

    # Increase resolution for better contour plot
    N = 300
    f = scipy.interpolate.interp2d(sigma, sigma0, ell, kind='cubic')
    sigma_fine = numpy.linspace(sigma[0], sigma[-1], N)
    sigma0_fine = numpy.linspace(sigma0[0], sigma0[-1], N)
    x, y = numpy.meshgrid(sigma_fine, sigma0_fine)
    ell_fine = f(sigma_fine, sigma0_fine)

    # We will plot the difference of max of ell to ell, called z
    # max_ell = numpy.abs(numpy.max(ell_fine))
    # z = max_ell - ell_fine
    z = ell_fine
    # x, y = numpy.meshgrid(sigma, sigma0)
    # z = ell

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
    cbar.ax.set_ylabel(r'$\ell(\sigma, \sigma_0)$')
    # c.set_clim(0, clim)
    # cbar.set_ticks([0,0.3,0.6,0.9,1])

    # Find max for each fixed sigma
    opt_sigma01 = numpy.zeros((sigma_fine.size, ), dtype=float)
    opt_ell1 = numpy.zeros((sigma_fine.size, ), dtype=float)
    opt_ell1[:] = numpy.nan
    for j in range(sigma_fine.size):
        if numpy.all(numpy.isnan(ell_fine[:, j])):
            continue
        max_index = numpy.nanargmax(ell_fine[:, j])
        opt_sigma01[j] = sigma_fine[max_index]
        opt_ell1[j] = ell_fine[max_index, j]
    ax[0].plot(sigma_fine, opt_sigma01, color='red',
               label=r'$\hat{\sigma}_0(\sigma)$')
    ax[1].plot(sigma_fine, opt_ell1, color='red')

    # Find max for each fixed sigma0
    opt_sigma2 = numpy.zeros((sigma0_fine.size, ), dtype=float)
    opt_ell2 = numpy.zeros((sigma0_fine.size, ), dtype=float)
    opt_ell2[:] = numpy.nan
    for i in range(sigma0_fine.size):
        if numpy.all(numpy.isnan(ell_fine[i, :])):
            continue
        max_index = numpy.nanargmax(ell_fine[i, :])
        opt_sigma2[i] = sigma0_fine[max_index]
        opt_ell2[i] = ell_fine[i, max_index]
    ax[0].plot(opt_sigma2, sigma0_fine, color='black',
               label=r'$\hat{\sigma}(\sigma_0)$')
    ax[2].plot(sigma0_fine, opt_ell2, color='black')

    # Plot max of the whole 2D array
    max_indices = numpy.unravel_index(numpy.nanargmax(ell_fine),
                                      ell_fine.shape)
    opt_sigma0 = sigma0_fine[max_indices[0]]
    opt_sigma = sigma_fine[max_indices[1]]
    opt_ell = ell_fine[max_indices[0], max_indices[1]]
    ax[0].plot(opt_sigma, opt_sigma0, 'o', color='red', markersize=6,
               label=r'$(\hat{\sigma}, \hat{\sigma}_0)$ (by brute force ' +
                     r'on grid)')
    ax[1].plot(opt_sigma, opt_ell, 'o', color='red',
               label=r'$\ell(\hat{\sigma}, \hat{\sigma}_0)$ by brute ' +
                     r'force on grid)')
    ax[2].plot(opt_sigma0, opt_ell, 'o', color='black',
               label=r'$\ell(\hat{\sigma}, \hat{\sigma}_0)$ (by brute ' +
                     r'force on grid)')

    # Plot optimal point as found by the profile likelihood method
    ax[0].plot(optimal_sigma, optimal_sigma0, 'X', color='black',
               markersize=6,
               label=r'$\max_{\sigma, \sigma_0} \ell$ (by optimization)')
    ax[1].plot(optimal_sigma, optimal_ell, 'X', color='red',
               label=r'$\ell(\hat{\sigma}, \hat{\sigma}_0)$ (by ' +
                     r'optimization)')
    ax[2].plot(optimal_sigma0, optimal_ell, 'X', color='black',
               label=r'$\ell(\hat{\sigma}, \hat{\sigma}_0)$ (by ' +
                     r'optimization)')

    # Plot annotations
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[0].set_xlim([sigma[0], sigma[-1]])
    ax[1].set_xlim([sigma[0], sigma[-1]])
    ax[2].set_xlim([sigma0[0], sigma0[-1]])
    ax[0].set_ylim([sigma0[0], sigma0[-1]])
    # ax[0].set_xscale('log')
    # ax[1].set_xscale('log')
    # ax[2].set_xscale('log')
    # ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\sigma$')
    ax[1].set_xlabel(r'$\sigma$')
    ax[2].set_xlabel(r'$\sigma_0$')
    ax[0].set_ylabel(r'$\sigma_0$')
    ax[1].set_ylabel(r'$\ell(\sigma, \hat{\sigma}_0(\sigma))$')
    ax[2].set_ylabel(r'$\ell(\hat{\sigma}(\sigma_0), sigma_0)$')
    ax[0].set_title('Log likelihood function')
    ax[1].set_title(r'Log Likelihood profiled over $\sigma$ ')
    ax[2].set_title(r'Log likelihood profiled over $\sigma_0$')
    ax[1].grid(True)
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()
