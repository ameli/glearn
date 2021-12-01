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

from ..kernels cimport Kernel


# ============
# Declarations
# ============

# Compute dense correlation
cdef double compute_dense_correlation(
        const double[:] point1,
        const double[:] point2,
        const int dimension,
        const double[:] scale,
        Kernel kernel) nogil

# Compute dense correlation jacobian
cdef void compute_dense_correlation_jacobian(
        const double[:, ::1] points,
        const int dimension,
        const double[:] scale,
        Kernel kernel,
        double[:, :, ::1] correlation_matrix_jacobian,
        int i,
        int j) nogil

# Compute dense correlation hessian
cdef void compute_dense_correlation_hessian(
        const double[:, ::1] points,
        const int dimension,
        const double[:] scale,
        Kernel kernel,
        double[:, :, :, ::1] correlation_matrix_hessian,
        int i,
        int j) nogil
