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

from libc.math cimport fabs
from .kernel import Kernel
from .kernel cimport Kernel

__all__ = ['Linear']


# ======
# Linear
# ======

cdef class Linear(Kernel):
    """
    Linear kernel.

    The kernel object is used as input argument to the instants of
    :class:`glearn.Covariance` class.

    .. note::

        For the methods of this class, see the base class
        :class:`glearn.kernels.Kernel`.

    See Also
    --------

    glearn.Covariance

    Notes
    -----

    The exponential kernel is defined as

    .. math::

        k(x) =
        \\begin{cases}
        1 - x, & x < 1, \\\\
        0, & \\text{otherwise}.
        \\end{cases}

    The first derivative of the kernel is

    .. math::

        \\frac{\\mathrm{d} k(x)}{\\mathrm{d}x} =
        \\begin{cases}
        -1, & x < 1, \\\\
        0, & \\text{otherwise}.
        \\end{cases}

    and its second derivative is

    .. math::

        \\frac{\\mathrm{d} k(x)}{\\mathrm{d}x} = 0.

    Examples
    --------

    **Create Kernel Object:**

    .. code-block:: python

        >>> from glearn import kernels

        >>> # Create an exponential kernel
        >>> kernel = kernels.Linear()

        >>> # Evaluate kernel at the point x=0.5
        >>> x = 0.5
        >>> kernel(x)
        0.5

        >>> # Evaluate first derivative of kernel at the point x=0.5
        >>> kernel(x, derivative=1)
        -1

        >>> # Evaluate second derivative of kernel at the point x=0.5
        >>> kernel(x, derivative=2)
        0

        >>> # Plot kernel and its first and second derivative
        >>> kernel.plot()

    .. image:: ../_static/images/plots/kernel_linear.png
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

    # =========
    # cy kernel
    # =========

    cdef double cy_kernel(self, const double x) nogil:
        """
        Computes the linear correlation function for a given Euclidean
        distance of two spatial points.

        The Linear correlation function defined by

        .. math::

            K(x) = \\begin{cases}
                1 - x,  & 0 \\leq x \\leq 1, \\
                0,      & x > 1
            \\end{cases}

        :param x: The distance that represents the Euclidean distance between
            mutual points.
        :type x: ndarray

        :return: Linear correlation kernel
        :rtype: double
        """

        if fabs(x) <= 1:
            return 1.0 - x
        else:
            return 0

    # ==========================
    # cy kernel first derivative
    # ==========================

    cdef double cy_kernel_first_derivative(self, const double x) nogil:
        """
        First derivative of kernel.
        """

        if fabs(x) <= 1:
            return -1.0
        else:
            return 0

    # ===========================
    # cy kernel second derivative
    # ===========================

    cdef double cy_kernel_second_derivative(self, const double x) nogil:
        """
        Second derivative of kernel.
        """

        return 0.0
