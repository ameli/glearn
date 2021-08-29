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
