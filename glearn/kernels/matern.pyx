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

from scipy.special.cython_special cimport gamma
from special_functions cimport besselk
from libc.math cimport sqrt, exp, fabs, isnan, isinf
from libc.stdio cimport printf
from .kernel import Kernel
from .kernel cimport Kernel
import numpy

__all__ = ['Matern']


# ======
# Matern
# ======

cdef class Matern(Kernel):
    """
    Matern kernel.

    The kernel object is used as input argument to the instants of
    :class:`glearn.Covariance` class.

    .. note::

        For the methods of this class, see the base class
        :class:`glearn.kernels.Kernel`.

    Parameters
    ----------

    nu : float, default=0.5
        The parameter :math:`\\nu` of the Matern function (see Notes below).

    See Also
    --------

    glearn.Covariance

    Notes
    -----

    **Matern Kernel Function:**

    Matern correlation (set :math:`\\nu` by ``nu``) is defined as

    .. math::

        k(x | \\nu) = \\frac{2^{1-\\nu}}{\\Gamma(\\nu)} \\left(
        \\sqrt{2 \\nu} x \\right)^{\\nu}
        K_{\\nu}\\left( \\sqrt{2 \\nu} x \\right),

    where :math:`K_{\\nu}` is the modified Bessel function of the
    second kind and :math:`\\Gamma` is the Gamma function. Both
    :math:`K_{\\nu}` and :math:`\\Gamma` are computed efficiently using the
    :func:`special_functions.besselk` and :func:`special_functions.lngamma`
    functions.

    The Matern kernel for specific values of :math:`\\nu` has simplified
    representation:

    * The Matern kernel with :math:`\\nu=\\frac{1}{2}` is equivalent to the
      exponential kernel, see :class:`glearn.kernels.Exponential`.

    * :math:`\\nu = \\infty` is equivalent to the square exponential kernel,
      see :class:`glearn.kernels.SquareExponential`. If :math:`\\nu > 100`,
      it is assumed that :math:`\\nu` is infinity.

    * If :math:`\\nu = \\frac{3}{2}` the following expression of Matern kernel
      is used:

      .. math::

          k(x | \\textstyle{\\frac{3}{2}}) =
          \\left(1+ \\sqrt{3} x \\right) e^{-\\sqrt{3} x}.

    * If :math:`\\nu = \\frac{5}{2}`, the Matern kernel is computed with:

      .. math::

          k(x | \\textstyle{\\frac{5}{2}}) =
          \\left(1+ \\sqrt{5} x + \\frac{5}{3} x^2 \\right) e^{-\\sqrt{5} x}.

    **First Derivative:**

    The first derivative of the kernel is computed as follows:

    .. math::

        \\frac{\\mathrm{d} k(x)}{\\mathrm{d}x} =
        c z^{\\nu - 1} \\left( \\nu K_{\\nu}(z) + z K'_{\\nu}(z) \\right),

    where :math:`K'_{\\nu}(z)` is the first derivative of the modified Bessel
    function of the second kind with respect to :math:`z`, and

    .. math::

        z =
        \\begin{cases}
            \\sqrt{2 \\nu} \\epsilon, & \\vert x \\vert < \\epsilon, \\\\
            \\sqrt{2 \\nu} x,         & \\vert x \\vert \\geq \\epsilon,
        \\end{cases}

    and

    .. math::

        c = \\frac{2^{1-\\nu}}{\\Gamma(\\nu)} \\sqrt{2 \\nu}.

    **Second Derivative:**

    The second derivative of the kernel is computed as follows:

    .. math::

        \\frac{\\mathrm{d} k(x)}{\\mathrm{d}x} =
        c z^{\\nu-2} \\left( \\nu (\\nu-1) K_{\\nu}(z) +
        2 z K'_{\\nu}(z) + z^2 K''_{\\nu}(z) \\right),

    where :math:`K''_{\\nu}(z)` is the second derivative of the modified Bessel
    function of the second kind with respect to :math:`z`.

    Examples
    --------

    **Create Kernel Object:**

    .. code-block:: python

        >>> from glearn import kernels

        >>> # Create an exponential kernel
        >>> kernel = kernels.Matern(nu=1.5)

        >>> # Evaluate kernel at the point x=0.5
        >>> x = 0.5
        >>> kernel(x)
        0.7848876539574506

        >>> # Evaluate first derivative of kernel at the point x=0.5
        >>> kernel(x, derivative=1)
        0.6309300390811722

        >>> # Evaluate second derivative of kernel at the point x=0.5
        >>> kernel(x, derivative=2)
        0.16905719445233686

        >>> # Plot kernel and its first and second derivative
        >>> kernel.plot()

    .. image:: ../_static/images/plots/kernel_matern.png
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

    def __cinit__(self, nu=0.5):
        """
        """

        if not isinstance(nu, (int, numpy.integer, float)):
            raise ValueError('"nu" should be a float or int type.')

        self.nu = nu

    # ==============
    # get parameters
    # ==============

    def get_parameters(self):
        """
        """

        parameters = [self.nu]
        return parameters

    # =========
    # cy kernel
    # =========

    cdef double cy_kernel(self, const double x) nogil:
        """
        Computes the Matern class correlation function for a given Euclidean
        distance of two spatial points.

        The Matern correlation function defined by

        .. math::

            K(x,\\nu) =
                \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}
                \\left( \\sqrt{2 \\nu} x \\right)
                K_{\\nu}\\left(\\sqrt{2 \\nu} x \\right)

        where

            * :math:`\\Gamma` is the Gamma function,
            * :math:`\\nu` is the smoothness parameter.
            * :math:`K_{\\nu}` is the modified Bessel function of the second
              kind of order :math:`\\nu`

        .. warning::

            When the distance :math:`\\| \\boldsymbol{x} -
            \\boldsymbol{x}' \\|` is zero, the correlation function produces
            :math:`\\frac{0}{0}` but in the limit, the correlation function is
            :math:`1`. If the distance is not exactly zero, but close to zero,
            this function might produce unstable results.

        :param x: The distance that represents the Euclidean distance between
            mutual points.
        :type x: ndarray

        :return: Matern correlation kernel
        :rtype: double
        """

        cdef double k
        cdef double epsilon = 1e-8

        if x == 0:
            k = 1.0
        else:
            if self.nu == 0.5:
                k = exp(-x)

            elif self.nu == 1.5:
                k = (1.0 + sqrt(3.0) * x) * exp(-sqrt(3.0) * x)

            elif self.nu == 2.5:
                k = (1.0 + sqrt(5.0) * x + (5.0 / 3.0) * (x**2)) * \
                        exp(-sqrt(5.0) * x)

            elif self.nu < 100:

                if fabs(x) < epsilon:
                    k = 1.0
                else:
                    k = ((2.0**(1.0-self.nu)) / gamma(self.nu)) * \
                            ((sqrt(2.0*self.nu) * x)**self.nu) * \
                            besselk(self.nu, sqrt(2.0*self.nu)*x, 0)

            else:
                # For nu > 100, assume nu is Inf. In this case, Matern function
                # approaches Gaussian kernel
                k = exp(-0.5*x**2)

            if isnan(k):
                printf('Matern kernel returned nan for the input x=' +
                       '%f and the parameter nu=%f.\n', x, self.nu)
            if isinf(k):
                printf('Matern kernel returned inf for the input x=' +
                       '%f and the parameter nu=%f.\n', x, self.nu)

        return k

    # ==========================
    # cy kernel first derivative
    # ==========================

    cdef double cy_kernel_first_derivative(self, const double x) nogil:
        """
        Computes the Matern class correlation function for a given Euclidean
        distance of two spatial points.

        The Matern correlation function defined by

        .. math::
            K(\\boldsymbol{x},\\boldsymbol{x}'|\\rho,\\nu) =
                \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}
                \\left( \\sqrt{2 \\nu} \\frac{\\| \\boldsymbol{x} -
                \\boldsymbol{x}' \\|}{\\rho} \\right)
                K_{\\nu}\\left(\\sqrt{2 \\nu}  \\frac{\\|\\boldsymbol{x} -
                \\boldsymbol{x}' \\|}{\\rho} \\right)

        where

            * :math:`\\rho` is the correlation scale of the function,
            * :math:`\\Gamma` is the Gamma function,
            * :math:`\\| \\cdot \\|` is the Euclidean distance,
            * :math:`\\nu` is the smoothness parameter.
            * :math:`K_{\\nu}` is the modified Bessel function of the second
              kind of order :math:`\\nu`

        .. warning::

            When the distance :math:`\\| \\boldsymbol{x} -
            \\boldsymbol{x}' \\|` is zero, the correlation function produces
            :math:`\\frac{0}{0}` but in the limit, the correlation function is
            :math:`1`. If the distance is not exactly zero, but close to zero,
            this function might produce unstable results.

        In this function, it is assumed that :math:`\\nu = \\frac{5}{2}`, and
        the Matern correlation in this case can be represented by:

        .. math::
            K(\\boldsymbol{x},\\boldsymbol{x}'|\\rho,\\nu) =
            \\left( 1 + \\sqrt{5} \\frac{\\| \\boldsymbol{x} -
            \\boldsymbol{x}'\\|}{\\rho} + \\frac{5}{3} \\frac{\\|
            \\boldsymbol{x} - \\boldsymbol{x}'\\|^2}{\\rho^2} \\right)
            \\exp \\left( -\\sqrt{5} \\frac{\\| \\boldsymbol{x} -
            \\boldsymbol{x}'\\|}{\\rho} \\right)

        :param x: The distance  that represents the Euclidean distance between
            mutual points.
        :type x: ndarray

        :return: Matern correlation kernel
        :rtype: double
        """

        cdef double k
        cdef double y
        cdef double epsilon = 1e-8
        cdef double c

        if self.nu == 0.5:
            dk = -exp(-x)

        elif self.nu == 1.5:
            dk = -3.0 * x * exp(-sqrt(3.0) * x)

        elif self.nu == 2.5:
            dk = -(5.0/3.0) * (x + sqrt(5.0) * x**2) * \
                    exp(-sqrt(5.0) * x)

        elif self.nu < 100:

            # Handle singular point of bessel function when x is close to zero
            if fabs(x) < epsilon and self.nu >= 1:
                dk = 0

            else:
                if fabs(x) < epsilon:
                    y = sqrt(2.0*self.nu) * epsilon
                else:
                    y = sqrt(2.0*self.nu) * x

                c = ((2.0**(1.0-self.nu)) / gamma(self.nu)) * sqrt(2.0*self.nu)
                dk = c * (y**(self.nu-1.0)) * \
                    (self.nu * besselk(self.nu, y, 0) +
                     y * besselk(self.nu, y, 1))

        else:
            # For nu > 100, assume nu is Inf. In this case, Matern function
            # approaches Gaussian kernel
            dk = -x * exp(-0.5*x**2)

        if isnan(dk):
            printf('Matern kernel first derivative returned nan for the ' +
                   'input x=%f and the parameter nu=%f.\n', x, self.nu)
        if isinf(dk):
            printf('Matern kernel first derivative returned inf for the ' +
                   'input  x= %f and the parameter nu=%f.\n', x, self.nu)

        return dk

    # ===========================
    # cy kernel second derivative
    # ===========================

    cdef double cy_kernel_second_derivative(self, const double x) nogil:
        """
        Computes the Matern class correlation function for a given Euclidean
        distance of two spatial points.

        The Matern correlation function defined by

        .. math::
            K(\\boldsymbol{x},\\boldsymbol{x}'|\\rho,\\nu) =
                \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}
                \\left( \\sqrt{2 \\nu} \\frac{\\| \\boldsymbol{x} -
                \\boldsymbol{x}' \\|}{\\rho} \\right)
                K_{\\nu}\\left(\\sqrt{2 \\nu}  \\frac{\\|\\boldsymbol{x} -
                \\boldsymbol{x}' \\|}{\\rho} \\right)

        where

            * :math:`\\rho` is the correlation scale of the function,
            * :math:`\\Gamma` is the Gamma function,
            * :math:`\\| \\cdot \\|` is the Euclidean distance,
            * :math:`\\nu` is the smoothness parameter.
            * :math:`K_{\\nu}` is the modified Bessel function of the second
              kind of order :math:`\\nu`

        .. warning::

            When the distance :math:`\\| \\boldsymbol{x} -
            \\boldsymbol{x}' \\|` is zero, the correlation function produces
            :math:`\\frac{0}{0}` but in the limit, the correlation function is
            :math:`1`. If the distance is not exactly zero, but close to zero,
            this function might produce unstable results.

        In this function, it is assumed that :math:`\\nu = \\frac{5}{2}`, and
        the Matern correlation in this case can be represented by:

        .. math::
            K(\\boldsymbol{x},\\boldsymbol{x}'|\\rho,\\nu) =
            \\left( 1 + \\sqrt{5} \\frac{\\| \\boldsymbol{x} -
            \\boldsymbol{x}'\\|}{\\rho} + \\frac{5}{3} \\frac{\\|
            \\boldsymbol{x} - \\boldsymbol{x}'\\|^2}{\\rho^2} \\right)
            \\exp \\left( -\\sqrt{5} \\frac{\\| \\boldsymbol{x} -
            \\boldsymbol{x}'\\|}{\\rho} \\right)

        :param x: The distance  that represents the Euclidean distance between
            mutual points.
        :type x: ndarray

        :return: Matern correlation kernel
        :rtype: double
        """

        cdef double k
        cdef double y
        cdef double epsilon = 1e-8
        cdef double c

        if self.nu == 0.5:
            d2k = exp(-x)

        elif self.nu == 1.5:
            d2k = -3.0 * (1.0 - sqrt(3.0) * x) * exp(-sqrt(3.0) * x)

        elif self.nu == 2.5:
            d2k = -(5.0/3.0) * (1.0 + sqrt(5.0)*x - 5.0*x**2) * \
                    exp(-sqrt(5.0) * x)

        elif self.nu < 100:

            # Handle singular point of bessel function when x is close to zero
            if fabs(x) < epsilon and self.nu >= 1:
                d2k = 0
            else:
                if fabs(x) < epsilon:
                    y = sqrt(2.0*self.nu) * epsilon
                else:
                    y = sqrt(2.0*self.nu) * x

                c = ((2.0**(1.0-self.nu)) / gamma(self.nu)) * 2.0*self.nu
                d2k = c * (y**(self.nu-2.0)) * (
                        self.nu * (self.nu - 1.0) * besselk(self.nu, y, 0) +
                        2.0 * y * besselk(self.nu, y, 1) +
                        y**2 * besselk(self.nu, y, 2))

        else:
            # For nu > 100, assume nu is Inf. In this case, Matern function
            # approaches Gaussian kernel
            d2k = (x**2 - 1.0) * exp(-0.5*x**2)

        if isnan(d2k):
            printf('Matern kernel second derivative returned nan for the ' +
                   'input x=%f and the parameter nu=%f.\n', x, self.nu)
        if isinf(d2k):
            printf('Matern kernel second derivative returned inf for the ' +
                   'input  x= %f and the parameter nu=%f.\n', x, self.nu)

        return d2k
