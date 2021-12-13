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
from .._utilities.plot_utilities import *                    # noqa: F401, F403
from .._utilities.plot_utilities import load_plot_settings, plt, \
    show_or_save_plot


# =====
# Prior
# =====

class Prior(object):
    """
    Base class for prior distributions.
    """

    def __init__(self, half=False):
        """
        Initialization.
        """

        # When True, derivatives of the pdf are taken w.r.t the logarithm of
        # the input hyperparameter. Default is True, but the Posterior class
        # can overwrite this attribute:
        self.use_log_scale = True

        # Using half distribution
        self.half = half

    # ===================
    # scale to hyperparam
    # ===================

    def _scale_to_hyperparam(self, scale):
        """
        Sets hyperparam from scale. ``scale`` is always given with no log-scale
        If self.use_log_eta is True, hyperparam is set as log10 of scale,
        otherwise, just as scale.
        """

        # If log scale is used, output hyperparam is log of scale.
        if self.use_log_scale:
            hyperparam = numpy.log10(numpy.abs(scale))
        else:
            hyperparam = numpy.abs(scale)

        return hyperparam

    # ===================
    # hyperparam to scale
    # ===================

    def _hyperparam_to_scale(self, hyperparam):
        """
        Sets scale from hyperparam. If self.use_log_scale is True, hyperparam
        is the log10 of scale, hence, 10**hyperparam is set to scale. If
        self.use_log_scale is False, hyperparam is directly set to scale.
        """

        # Convert to numpy array
        if numpy.isscalar(hyperparam):
            hyperparam = numpy.array([hyperparam], dtype=float)
        elif isinstance(hyperparam, list):
            hyperparam = numpy.array(hyperparam, dtype=float)

        # If log scale is used, input hyperparam is log of the scale.
        if self.use_log_scale:
            scale = 10.0**hyperparam
        else:
            scale = numpy.abs(hyperparam)

        return scale

    # =======
    # log pdf
    # =======

    def log_pdf(self, hyperparam):
        """
        Returns the log of probability distribution function.
        """

        # Convert hyperparam from log to non-log.
        scale = self._hyperparam_to_scale(hyperparam)

        if self.half and any(scale < 0.0):
            raise ValueError('"hyperparam" cannot be negative for ' +
                             'half-distributions.')

        # Call derived class's method
        pdf_ = self.pdf(scale)

        # Take log of the product of all distributions
        log_pdf_ = numpy.sum(numpy.log(pdf_))

        if self.half:
            log_pdf_ += numpy.log(2.0)

        return log_pdf_

    # ================
    # log pdf jacobian
    # ================

    def log_pdf_jacobian(self, hyperparam):
        """
        Returns the Jacobian of prior probability density function either
        with respect to the hyperparam or the log of hyperparam.
        """

        # Convert hyperparam from log to non-log (if needed)
        scale = self._hyperparam_to_scale(hyperparam)

        if self.half and any(scale < 0.0):
            raise ValueError('"hyperparam" cannot be negative for ' +
                             'half-distributions.')

        # Call derived class's method
        pdf_ = self.pdf(scale)
        pdf_jacobian_ = self.pdf_jacobian(scale)

        # Take log of the pdf
        log_pdf_jacobian_ = numpy.zeros((scale.size, ), dtype=float)
        for i in range(scale.size):
            log_pdf_jacobian_[i] = pdf_jacobian_[i] / pdf_[i]

        # Convert derivative w.r.t log of scale
        if self.use_log_scale:
            for i in range(scale.size):
                log_pdf_jacobian_[i] = log_pdf_jacobian_[i] * scale[i] * \
                        numpy.log(10.0)

        return log_pdf_jacobian_

    # ===============
    # log pdf hessian
    # ===============

    def log_pdf_hessian(self, hyperparam):
        """
        Returns the Hessian of prior probability density function either
        with respect to the hyperparam or the log of hyperparam.
        """

        # Convert hyperparam from log to non-log (if needed)
        scale = self._hyperparam_to_scale(hyperparam)

        if self.half and any(scale < 0.0):
            raise ValueError('"hyperparam" cannot be negative for ' +
                             'half-distributions.')

        # Call derived class's method
        pdf_ = self.pdf(scale)
        pdf_jacobian_ = self.pdf_jacobian(scale)
        pdf_hessian_ = self.pdf_hessian(scale)

        # Take log of the pdf
        log_pdf_hessian_ = numpy.zeros((scale.size, scale.size), dtype=float)
        for i in range(scale.size):
            log_pdf_hessian_[i, i] = (pdf_hessian_[i, i] / pdf_[i]) - \
                    (pdf_jacobian_[i] / pdf_[i])**2

        # Convert derivative w.r.t log of scale
        if self.use_log_scale:

            # To convert derivative to log scale, Jacobian is needed. Note:
            # The Jacobian itself is already converted to log scale.
            log_pdf_jacobian_ = self.log_pdf_jacobian(hyperparam)

            for p in range(scale.size):
                for q in range(scale.size):
                    if p == q:

                        # log_pdf_jacobian_ is already converted to log scale
                        log_pdf_hessian_[p, q] = log_pdf_hessian_[p, q] * \
                            scale[p]**2 * (numpy.log(10.0)**2) + \
                            log_pdf_jacobian_[p] * numpy.log(10.0)
                    else:
                        log_pdf_hessian_[p, q] = log_pdf_hessian_[p, q] * \
                            scale[p] * scale[q] * (numpy.log(10.0)**2)

        return log_pdf_hessian_

    # ====
    # plot
    # ====

    def plot(self, x_range=[0, 2], log_scale=False, compare_numerical=False):
        """
        Plots the distribution.
        """

        load_plot_settings()

        # Check range
        if not isinstance(x_range, (list, tuple)):
            raise TypeError('"x_range" should be a list or a tuple')
        elif len(x_range) != 2:
            raise ValueError('"x_range" should be 1d array of size 2.')
        elif x_range[0] >= x_range[1]:
            raise ValueError('"x_range[0]" should be less than "x_range[1]".')

        # Avoid plotting from origin in log-scale x-axis
        if log_scale and x_range[0] == 0.0:
            x_range[0] = numpy.min([(x_range[1] - x_range[0]) * 1e-2, 1e-2])

        # Abscissa
        num_points = 200
        if log_scale:
            x = numpy.logspace(numpy.log10(x_range[0]),
                               numpy.log10(x_range[1]), num_points)
        else:
            x = numpy.linspace(x_range[0], x_range[1], num_points)

        # Convert x to log of x (if enabled by log_scale)
        if log_scale:
            hyperparam = numpy.log10(numpy.abs(x))
        else:
            # Note: don't use abs(x), for some distributions, x may be negative
            hyperparam = x

        # Allocate outputs
        d0f = numpy.zeros_like(hyperparam)
        d1f = numpy.zeros_like(hyperparam)
        d2f = numpy.zeros_like(hyperparam)

        # Generate distribution and its derivatives
        for i in range(hyperparam.size):

            # Compute the pdf and its first and second derivative
            if log_scale:
                d0f[i] = self.log_pdf(hyperparam[i])
                d1f[i] = self.log_pdf_jacobian(hyperparam[i])
                d2f[i] = self.log_pdf_hessian(hyperparam[i])
            else:
                d0f[i] = self.pdf(hyperparam[i])
                d1f[i] = self.pdf_jacobian(hyperparam[i])
                d2f[i] = self.pdf_hessian(hyperparam[i])

        # Compare analytic derivative with numerical derivative
        if compare_numerical:
            d1f_num = numpy.zeros_like(hyperparam.size-2)
            d2f_num = numpy.zeros_like(hyperparam.size-4)

            d1f_num = (d0f[2:] - d0f[:-2]) / (hyperparam[2:] - hyperparam[:-2])
            d2f_num = (d1f_num[2:] - d1f_num[:-2]) / \
                (hyperparam[3:-1] - hyperparam[1:-3])

        # Plotting
        fig, ax = plt.subplots(ncols=3, figsize=(17, 5))
        ax[0].plot(x, d0f, color='black')
        ax[1].plot(x, d1f, color='black', label='analytic')
        ax[2].plot(x, d2f, color='black', label='analytic')
        ax[0].set_xlabel(r'$x$')
        ax[1].set_xlabel(r'$x$')
        ax[2].set_xlabel(r'$x$')

        if compare_numerical:
            ax[1].plot(x[1:-1], d1f_num, '--', color='black',
                       label='numerical')
            ax[2].plot(x[2:-2], d2f_num, '--', color='black',
                       label='numerical')
            ax[1].legend()
            ax[2].legend()

        if log_scale:
            ax[0].set_ylabel(r'$\ln p(x)$')
        else:
            ax[0].set_ylabel(r'$p(x)$')

        if log_scale:
            ax[1].set_ylabel(r'$\frac{\mathrm{d}\ln p(x)}{\mathrm{d}(\ln x)}$')
        else:
            ax[1].set_ylabel(r'$\frac{\mathrm{d}p(x)}{\mathrm{d}x}$')

        if log_scale:
            ax[2].set_ylabel(r'$\frac{\mathrm{d}^2\ln p(x)}{\mathrm{d} ' +
                             r'(\ln x)^2}$')
        else:
            ax[2].set_ylabel(r'$\frac{\mathrm{d}^2 p(x)}{\mathrm{d}x^2}$')

        ax[0].set_title('Probability distribution')
        ax[1].set_title('First derivative of probability distribution')
        ax[2].set_title('Second derivative of probability distribution')
        ax[0].set_xlim([x_range[0], x_range[1]])
        ax[1].set_xlim([x_range[0], x_range[1]])
        ax[2].set_xlim([x_range[0], x_range[1]])
        ax[0].grid(True, which='both')
        ax[1].grid(True, which='both')
        ax[2].grid(True, which='both')

        if log_scale:
            ax[0].set_xscale('log', base=10)
            ax[1].set_xscale('log', base=10)
            ax[2].set_xscale('log', base=10)

        plt.tight_layout()
        show_or_save_plot(plt, 'prior', transparent_background=True)
