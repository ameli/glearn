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

from libc.math cimport exp
from .kernel import Kernel
from .kernel cimport Kernel
import numpy

__all__ = ['RationalQuadratic']


# ==================
# Rational Quadratic
# ==================

cdef class RationalQuadratic(Kernel):
    """
    Rational quadratic kernel.

    The kernel object is used as input argument to the instants of
    :class:`glearn.Covariance` class.

    .. note::

        For the methods of this class, see the base class
        :class:`glearn.kernels.Kernel`.

    Parameters
    ----------

    alpha : float, default=2.0
        The parameter :math:`\\alpha` of the rational quadratic function (see
        Notes below).

    See Also
    --------

    glearn.Covariance

    Notes
    -----

    The exponential kernel is defined as

    .. math::

        k(x) = \\left( 1 + \\frac{x^2}{2 \\alpha} \\right)^{-\\alpha}.

    The first derivative of the kernel is

    .. math::

        \\frac{\\mathrm{d} k(x)}{\\mathrm{d}x} = -x k(x)^{-1-\\alpha},

    and its second derivative is

    .. math::

        \\frac{\\mathrm{d} k(x)}{\\mathrm{d}x} =
        x^2 (1 + \\alpha^{-1}) k(x)^{-2-\\alpha} - k(x)^{-1-\\alpha}.

    Examples
    --------

    **Create Kernel Object:**

    .. code-block:: python

        >>> from glearn import kernels

        >>> # Create an exponential kernel
        >>> kernel = kernels.RationalQuadratic(alpha=2.0)

        >>> # Evaluate kernel at the point x=0.5
        >>> x = 0.5
        >>> kernel(x)
        0.8858131487889274

        >>> # Evaluate first derivative of kernel at the point x=0.5
        >>> kernel(x, derivative=1)
        0.416853246488907

        >>> # Evaluate second derivative of kernel at the point x=0.5
        >>> kernel(x, derivative=2)
        0.416853246488907

        >>> # Plot kernel and its first and second derivative
        >>> kernel.plot()

    .. image:: ../_static/images/plots/kernel_rational_quadratic.png
        :align: center
        :width: 100%
        :class: custom-dark

    **Where to Use Kernel Object:**

    Use the kernel object to define a covariance object:

    .. code-block:: python
        :emphasize-lines: 7

        >>> # Generate a set of sample points
        >>> from glearn.sample_data import generate_points
        >>> points = generate_points(num_points=50)

        >>> # Create covariance object of the points with the above kernel
        >>> from glearn import covariance
        >>> cov = glearn.Covariance(points, kernel=kernel)
    """

    # =====
    # cinit
    # =====

    def __cinit__(self, alpha=2.0):
        """
        """

        if not isinstance(alpha, (int, numpy.integer, float)):
            raise ValueError('"alpha" should be a float or int type.')

        self.alpha = alpha

    # ==============
    # get parameters
    # ==============

    def get_parameters(self):
        """
        """

        parameters = [self.alpha]
        return parameters

    # =========
    # cy kernel
    # =========

    cdef double cy_kernel(self, const double x) nogil:
        """
        Computes the rational quadratic correlation function for a given
        Euclidean distance of two spatial points.

        The Exponential correlation function defined by

        .. math::

            k(x) = \\left( 1 + \\frac{x^2}{2 \\alpha} \\right)^{-\\alpha}.

        :param x: The distance that represents the Euclidean distance between
            mutual points.
        :type x: ndarray

        :return: Square exponential correlation kernel
        :rtype: double
        """

        cdef double k = 1.0 / (1.0 + x**2 / (2.0 * self.alpha))

        if self.alpha != 1.0:
            k = k**(self.alpha)

        return k

    # ==========================
    # cy kernel first derivative
    # ==========================

    cdef double cy_kernel_first_derivative(self, const double x) nogil:
        """
        First derivative of kernel.
        """

        cdef double k = 1.0 / (1.0 + x**2 / (2.0 * self.alpha))
        return -x * k**(self.alpha+1.0)

    # ===========================
    # cy kernel second derivative
    # ===========================

    cdef double cy_kernel_second_derivative(self, const double x) nogil:
        """
        Second derivative of kernel.
        """

        cdef double k = 1.0 / (1.0 + x**2 / (2.0 * self.alpha))
        return x**2 * (1.0 + 1.0/self.alpha) * k**(self.alpha+2.0) - \
            k**(self.alpha+1.0)
