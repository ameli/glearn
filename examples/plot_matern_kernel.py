#! /usr/bin/env python

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

import sys
import numpy
import scipy.special
from _utilities.plot_utilities import *                      # noqa: F401, F403
from _utilities.plot_utilities import load_plot_settings, save_plot, plt


# =============
# matern kernel
# =============

def _matern_kernel(nu, x):
    """
    Matern correlation Kernel on variable ``x`` and parameter ``nu``. If ``nu``
    is infinity, the Matern correlation kernel becomes the Gaussian kernel.

    When ``x = 0``, the kernel should return ``1``. In this case, to avoid
    multiplication of zero by infinity, we exclude indices of ``x`` that
    ``x == 0``.
    """

    if numpy.isinf(nu):

        # Gassian Kernel
        y = numpy.exp(-0.5*x**2)

    else:

        # If x is zero, avoid multiplication of zero nu infinity
        zero_index = numpy.where(x == 0)[0][:]

        # Alter zeros in x to a dummy value. We will return it back.
        z = numpy.copy(x)
        z[zero_index] = 1

        y = ((2**(1-nu))/scipy.special.gamma(nu)) * \
            ((numpy.sqrt(2.0*nu)*z)**nu) * \
            scipy.special.kv(nu, numpy.sqrt(2.0*nu)*z)

        # Correlation at x = 0 is 1.
        y[zero_index] = 1

    return y


# ====
# main
# ====

def main(test=False):
    """
    Set ``plot_errors`` to True to plot errors besides the Matern correlation
    kernel.The errors are the difference between Matern kernel and Gaussian
    kernel. The Gaussian kernel is essentially the Matern kernel for the
    parameter ``nu = infinity``. The purpose of the plot are to show at
    ``nu > 25``, the Matern kernel is almost the same as the Gaussian kernel
    with less than ``1`` percent difference error.
    """

    # Load plot settings
    sns = load_plot_settings()

    # Set this to True for error plots
    plot_errors = False

    if plot_errors:
        num_cols = 2
        fig_size = (9.7, 3.8)
    else:
        num_cols = 1
        fig_size = (5, 3.5)

    fig, ax = plt.subplots(ncols=num_cols, figsize=fig_size)
    if not plot_errors:
        ax = [ax] # to use ax[0]

    # Set nu
    nu = [0.1, 0.5, 1.0, 3.2, 25]
    nu_labels = ['0.1', '0.5', '1.0', '3.2', '25']

    colors = sns.color_palette("OrRd_d", len(nu))[::-1]
    x = numpy.linspace(0, 4, 1000)

    for i in range(len(nu)):
        ax[0].plot(x, _matern_kernel(nu[i], x),
                   label=r'$\nu = %s$'%(nu_labels[i]), color=colors[i])

        if plot_errors:
            ax[1].plot(x,
                       _matern_kernel(numpy.inf, x) - _matern_kernel(nu[i], x),
                       label=r'$\nu = %s$'%(nu_labels[i]), color=colors[i])

    # Gaussian kernel at nu = infinity
    ax[0].plot(x, _matern_kernel(numpy.inf, x), label=r'$\nu = \infty$',
               color='black')

    ax[0].legend(frameon=False)
    ax[0].set_xlim([x[0], x[-1]])
    ax[0].set_ylim([0, 1])
    ax[0].set_xticks(numpy.arange(0, 4.01, 1))
    ax[0].set_yticks(numpy.arange(0, 1.01, 0.5))
    ax[0].set_xlabel(r'$r$')
    ax[0].set_ylabel(r'$K(r|\nu)$')
    ax[0].set_title(r'Mat\'{e}rn Correlation Kernel')

    if plot_errors:
        ax[1].legend(frameon=False)
        ax[1].set_xlim([x[0], x[-1]])
        ax[1].set_ylim([-0.1, 0.7])
        ax[1].set_xticks(numpy.arange(0, 4.01, 1))
        ax[1].set_yticks([-0.1, 0, 0.1, 0.7])
        ax[1].set_xlabel(r'$r$')
        ax[1].set_ylabel(r'$K(r|\infty) - K(r|\nu)$')
        ax[1].set_title(r'Difference of Gaussian and Mat\'{e}rn kernels')

    plt.tight_layout()

    # Save plot
    filename = 'matern_kernel'
    if test:
        filename = "test_" + filename
    save_plot(plt, filename, transparent_background=False, pad_inches=0)

    # If no display backend is enabled, do not plot in the interactive mode
    if (not test) and (matplotlib.get_backend() != 'agg'):
        plt.show()


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(main())
