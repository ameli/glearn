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

            K(x) = \\exp(-x^2 / 2)

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
