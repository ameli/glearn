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

__all__ = ['Exponental']


# ===========
# Exponential
# ===========

cdef class Exponential(Kernel):
    """
    """

    # =========
    # cy kernel
    # =========

    cdef double cy_kernel(self, const double x) nogil:
        """
        Computes the exponential correlation function for a given Euclidean
        distance of two spatial points.

        The Exponential correlation function defined by

        .. math::

            K(x) = \\exp(-x)

        :param x: The distance that represents the Euclidean distance between
            mutual points.
        :type x: ndarray

        :return: Exponential correlation kernel
        :rtype: double
        """

        return exp(-x)

    # ==========================
    # cy kernel first derivative
    # ==========================

    cdef double cy_kernel_first_derivative(self, const double x) nogil:
        """
        First derivative of kernel.
        """

        return -exp(-x)

    # ===========================
    # cy kernel second derivative
    # ===========================

    cdef double cy_kernel_second_derivative(self, const double x) nogil:
        """
        Second derivative of kernel.
        """

        return exp(-x)
