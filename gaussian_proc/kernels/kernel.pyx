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

from libc.stdio cimport printf
from libc.math cimport NAN


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
