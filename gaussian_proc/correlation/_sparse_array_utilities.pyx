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

from libc.stdlib cimport exit, malloc, free

__all__ = ['extend_arrays']


# =============
# extend arrays
# =============

cdef void extend_arrays(
        long[:] max_nnz,
        long[:] nnz,
        long** pp_matrix_row_indices,
        long** pp_matrix_column_indices,
        double** pp_matrix_data) nogil:
    """
    Extends the size of arrays by factor of two.
    """

    # Extended arrays have double the size of the previous arrays
    cdef long ext_nnz = max_nnz[0] * 2

    # Allocate extended arrays
    cdef long* ext_matrix_row_indices = <long*> malloc(ext_nnz * sizeof(long))
    cdef long* ext_matrix_column_indices = <long*> malloc(
        ext_nnz * sizeof(long))
    cdef double* ext_matrix_data = <double*> malloc(ext_nnz * sizeof(double))

    # Copy contents of previous arrays
    cdef int i
    for i in range(nnz[0]):
        ext_matrix_row_indices[i] = pp_matrix_row_indices[0][i]
        ext_matrix_column_indices[i] = pp_matrix_column_indices[0][i]
        ext_matrix_data[i] = pp_matrix_data[0][i]

    # Delete previous arrays
    free(pp_matrix_row_indices[0])
    free(pp_matrix_column_indices[0])
    free(pp_matrix_data[0])

    # Transfer pointers
    pp_matrix_row_indices[0] = ext_matrix_row_indices
    pp_matrix_column_indices[0] = ext_matrix_column_indices
    pp_matrix_data[0] = ext_matrix_data

    # Update size
    max_nnz[0] = ext_nnz
