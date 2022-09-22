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
        save_plot, show_or_save_plot
    plot_modules_exist = True
except ImportError:
    plot_modules_exist = False

__all__ = ['Kernel']


# ======
# Kernel
# ======

cdef class Kernel(object):
    """
    Base class of kernel functions.

    .. warning::

        This class is a base class and does not implement a kernel function.
        Use the derivative of this class instead.

    Methods
    -------

    __call__
    plot

    See Also
    --------

    glearn.kernels.Exponential
    glearn.kernels.SquareExponential
    glearn.kernels.Linear
    glearn.kernels.RationalQuadratic
    glearn.kernels.Matern

    Examples
    --------

    **Create Kernel Object:**

    .. code-block:: python

        >>> from glearn import kernels

        >>> # Create an exponential kernel
        >>> kernel = kernels.Exponential()

        >>> # Evaluate kernel at the point x=0.5
        >>> x = 0.5
        >>> kernel(x)
        0.6065306597126334

        >>> # Evaluate first derivative of kernel at the point x=0.5
        >>> kernel(x, derivarive=1)
        -0.6065306597126334

        >>> # Evaluate second derivative of kernel at the point x=0.5
        >>> kernel(x, derivarive=2)
        0.6065306597126334

        >>> # Plot kernel and its first and second derivative
        >>> kernel.plot()

    .. image:: ../_static/images/plots/kernel_exponential.png
        :align: center
        :width: 100%
        :class: custom-dark
    """

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

    # ========
    # __call__
    # ========

    def __call__(self, x, derivative=0):
        """
        Evaluate the kernel function or its derivatives.

        Parameters
        ----------

        x : float or array_like[float]
            Input points to the kernel function.

        derivative : int, default=0
            The order of the derivative of the kernel function. Zero means no
            derivative.

        Returns
        -------

        y : float or numpy.array[float]
            The value of the kernel function or its derivatives. The size of
            ``y`` is the same as the size of the input argument ``x``.

        Examples
        --------

        .. code-block:: python

            >>> from glearn import kernels

            >>> # Create an exponential kernel
            >>> kernel = kernels.Exponential()

            >>> # Evaluate kernel at the point x=0.5
            >>> x = 0.5
            >>> kernel(x)
            0.6065306597126334

            >>> # Evaluate first derivative of kernel at the point x=0.5
            >>> kernel(x, derivarive=1)
            -0.6065306597126334

            >>> # Evaluate second derivative of kernel at the point x=0.5
            >>> kernel(x, derivarive=2)
            0.6065306597126334
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

    def plot(self, compare_numerical=False, x_max=4.0, test=False):
        """
        Plot the kernel function and its first and second derivative.

        Parameters
        ----------

        compare_numerical : bool, default=False
            It `True`, it computes the derivatives of the kernel function and
            plots the numerical derivatives together with the exact values of
            the derivatives from analytical formula. This is used to validate
            the analytical formulas.

        x_max : float, default=4.0
            Maximum range in the abscissa in the plot.

        test : bool, default=False
            If `True`, this function is used for test purposes.

        Notes
        -----

        * If no graphical backend exists (such as running the code on a remote
          server or manually disabling the X11 backend), the plot will not be
          shown, rather, it will be saved as an ``svg`` file in the current
          directory.
        * If the executable ``latex`` is available on ``PATH``, the plot is
          rendered using :math:`\\rm\\LaTeX` and it may take slightly longer to
          produce the plot.
        * If :math:`\\rm\\LaTeX` is not installed, it uses any available
          San-Serif font to render the plot.

        To manually disable interactive plot display and save the plot as
        ``svg`` instead, add the following at the very beginning of your code
        before importing :mod:`glearn`:

        .. code-block:: python

            >>> import os
            >>> os.environ['GLEARN_NO_DISPLAY'] = 'True'

        Examples
        --------

        **Create Kernel Object:**

        .. code-block:: python

            >>> from glearn import kernels

            >>> # Create an exponential kernel
            >>> kernel = kernels.Exponential()

            >>> # Plot kernel and its first and second derivative
            >>> kernel.plot()

        .. image:: ../_static/images/plots/kernel_exponential.png
            :align: center
            :width: 100%
            :class: custom-dark
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
            d0y[i] = self.__call__(x[i], derivative=0)
            d1y[i] = self.__call__(x[i], derivative=1)
            d2y[i] = self.__call__(x[i], derivative=2)

        ax[0].plot(x, d0y, color='black')
        ax[1].plot(x, d1y, color='black', label='Analytic')
        ax[2].plot(x, d2y, color='black', label='Analytic')

        # Compare analytic derivative with numerical derivative
        if compare_numerical:
            d1y_num = (d0y[2:] - d0y[:n-2]) / (x[2:] - x[:n-2])
            d2y_num = (d1y_num[2:] - d1y_num[:n-4]) / (x[3:n-1] - x[1:n-3])
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

        if test:
            save_plot(plt, 'kernel', pdf=False, verbose=False)
        else:
            show_or_save_plot(plt, 'kernel', transparent_background=True)
