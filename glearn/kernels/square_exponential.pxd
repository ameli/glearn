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

from .kernel cimport Kernel


# ==================
# Square Exponential
# ==================

cdef class SquareExponential(Kernel):

    # Methods
    cdef double cy_kernel(self, const double x) nogil
    cdef double cy_kernel_first_derivative(self, const double x) nogil
    cdef double cy_kernel_second_derivative(self, const double x) nogil
