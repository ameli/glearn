# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import numpy
from libc.stdio cimport printf
from libc.math cimport NAN

try:
    from .._utilities.plot_utilities import matplotlib, plt
    from .._utilities.plot_utilities import load_plot_settings, \
        show_or_save_plot
    plot_modules_exist = True
except ImportError:
    plot_modules_exist = False


# ======
# Kernel
# ======

cdef class Kernel(object):

    # =========
    # cy kernel
    # =========

    cdef double cy_kernel(self, const double x) nogil:
        """
        """

        printf('Not Implemented ERROR: this method should only be called by ' +
               'a derived class.')
        return NAN

    # ==========================
    # cy kernel first derivative
    # ==========================

    cdef double cy_kernel_first_derivative(self, const double x) nogil:
        """
        """
        printf('Not Implemented ERROR: this method should only be called by ' +
               'a derived class.')
        return NAN

    # ===========================
    # cy kernel second derivative
    # ===========================

    cdef double cy_kernel_second_derivative(self, const double x) nogil:
        """
        """
        printf('Not Implemented ERROR: this method should only be called by ' +
               'a derived class.')
        return NAN

    # ======
    # kernel
    # ======

    def kernel(self, x, derivative=0):
        """
        A python wrapper for ``cy_kernel``.
        """

        if derivative == 0:
            k = self.cy_kernel(x)

        elif derivative == 1:
            k = self.cy_kernel_first_derivative(x)

        elif derivative == 2:
            k = self.cy_kernel_second_derivative(x)

        else:
            raise NotImplementedError('"derivative" can only be "0", "1", ' +
                                      'or "2"')

        return k

    # ====
    # plot
    # ====

    def plot(self, compare_numerical=False, x_max=4.0):
        """
        Plots the kernel function and its first and second derivative
        """

        # Load plot settings
        if plot_modules_exist:
            load_plot_settings()
        else:
            raise ImportError("Cannot load plot settings.")

        fig, ax = plt.subplots(ncols=3, figsize=(12.5, 4))

        x = numpy.linspace(0, x_max, 200)
        d0y = numpy.zeros_like(x)
        d1y = numpy.zeros_like(x)
        d2y = numpy.zeros_like(x)
        n = x.size

        for i in range(x.size):
            d0y[i] = self.kernel(x[i], derivative=0)
            d1y[i] = self.kernel(x[i], derivative=1)
            d2y[i] = self.kernel(x[i], derivative=2)

        ax[0].plot(x, d0y, color='black')
        ax[1].plot(x, d1y, color='black', label='Analytic')
        ax[2].plot(x, d2y, color='black', label='Analytic')

        # Compare analytic derivative with numerical derivative
        if compare_numerical:
            d1y_num = (d0y[2:] - d0y[:n-2]) / (x[2:] - x[:n-2])
            d2y_num = (d1y_num[2:] - d1y_num[:n-2]) / (x[3:n-1] - x[1:n-3])
            ax[1].plot(x[1:n-1], d1y_num, '--', color='black',
                       label='Numerical')
            ax[2].plot(x[2:n-2], d2y_num, '--', color='black',
                       label='Numerical')
            ax[1].legend()
            ax[2].legend()

        ax[0].set_ylim([0, 1])
        ax[0].set_xlim([x[0], x[n-1]])
        ax[1].set_xlim([x[0], x[n-1]])
        ax[2].set_xlim([x[0], x[n-1]])

        ax[0].set_xlabel(r'$x$')
        ax[1].set_xlabel(r'$x$')
        ax[2].set_xlabel(r'$x$')

        ax[0].set_ylabel(r'$k(x)$')
        ax[1].set_ylabel(r'$\frac{\mathrm{d} k(x)}{\mathrm{d}x}$')
        ax[2].set_ylabel(r'$\frac{\mathrm{d}^2 k(x)}{\mathrm{d}x^2}$')

        ax[0].set_title('Kernel')
        ax[1].set_title('Kernel First Derivative')
        ax[2].set_title('Kernel Second Derivative')

        ax[0].grid(True)
        ax[1].grid(True)
        ax[2].grid(True)

        plt.tight_layout()
        show_or_save_plot(plt, 'kernel', transparent_background=True)
