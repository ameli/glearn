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
