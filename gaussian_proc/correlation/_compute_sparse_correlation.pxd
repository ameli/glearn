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

# Compute sparse correlation
cdef double compute_sparse_correlation(
        const double[:] point1,
        const double[:] point2,
        const int dimension,
        const double[:] scale,
        Kernel kernel) nogil

# Compute sparse correlation jacobian
cdef void compute_sparse_correlation_jacobian(
        const double[:, ::1] points,
        const int dimension,
        const double[:] scale,
        Kernel kernel,
        int row,
        int index_pointer,
        int[:] matrix_column_indices,
        double[:, ::1] matrix_data) nogil

# Compute sparse correlation hessian
cdef void compute_sparse_correlation_hessian(
        const double[:, ::1] points,
        const int dimension,
        const double[:] scale,
        Kernel kernel,
        int row,
        int index_pointer,
        int[:] matrix_column_indices,
        double[:, :, ::1] matrix_data) nogil
