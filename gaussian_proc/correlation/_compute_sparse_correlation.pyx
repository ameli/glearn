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

from cython import boundscheck, wraparound
from ._euclidean_distance cimport euclidean_distance
from ..kernels cimport Kernel

__all__ = ['compute_sparse_correlation', 'compute_sparse_correlation_jacobian',
           'compute_sparse_correlation_hessian']


# ==========================
# compute sparse correlation
# ==========================

@boundscheck(False)
@wraparound(False)
cdef double compute_sparse_correlation(
        const double[:] point1,
        const double[:] point2,
        const int dimension,
        const double[:] scale,
        Kernel kernel) nogil:
    """
    Computes the i-j entry of the correlation matrix.
    """

    cdef double distance = euclidean_distance(point1, point2, scale, dimension)

    return kernel.cy_kernel(distance)


# ===================================
# compute sparse correlation jacobian
# ===================================

@boundscheck(False)
@wraparound(False)
cdef void compute_sparse_correlation_jacobian(
        const double[:, ::1] points,
        const int dimension,
        const double[:] scale,
        Kernel kernel,
        int row,
        int index_pointer,
        int[:] matrix_column_indices,
        double[:, ::1] matrix_data) nogil:
    """
    Computes the Jacobian of correlation with respect to scale
    parameters.
    """

    # Iterator for Jacobian derivative
    cdef int p

    # Find column of csr matrix
    cdef int column = matrix_column_indices[index_pointer]

    # The case at zero distance.
    if row == column:
        for p in range(dimension):
            matrix_data[p, index_pointer] = 0.0
        return

    cdef double distance = euclidean_distance(
            points[row][:], points[column][:], scale, dimension)

    cdef double d1_kernel = kernel.cy_kernel_first_derivative(distance)

    # Derivative of distance w.r.t one of the components of scale
    cdef double d1_distance

    for p in range(dimension):

        # derivative of distance w.r.t the p-th component of scale
        d1_distance = -(points[row, p] - points[column, p])**2 / \
            (distance * scale[p]**3)

        # Derivative of correlation
        matrix_data[p, index_pointer] = d1_kernel * d1_distance


# ==================================
# compute sparse correlation hessian
# ==================================

@boundscheck(False)
@wraparound(False)
cdef void compute_sparse_correlation_hessian(
        const double[:, ::1] points,
        const int dimension,
        const double[:] scale,
        Kernel kernel,
        int row,
        int index_pointer,
        int[:] matrix_column_indices,
        double[:, :, ::1] matrix_data) nogil:
    """
    Computes the Hessian of correlation with respect to scale
    parameters.
    """

    cdef int p, q

    # Find column of csr matrix
    cdef int column = matrix_column_indices[index_pointer]

    # The case at zero distance
    if row == column:
        for p in range(dimension):
            for q in range(dimension):
                matrix_data[p, q, index_pointer] = 0.0
        return

    cdef double distance = euclidean_distance(
            points[row][:], points[column][:], scale, dimension)

    cdef double d1_kernel = kernel.cy_kernel_first_derivative(distance)
    cdef double d2_kernel = kernel.cy_kernel_second_derivative(distance)

    # Derivative of distance w.r.t one of the components of scale
    cdef double dp_distance
    cdef double dq_distance
    cdef double dpq_distance

    for p in range(dimension):
        for q in range(p, dimension):

            # derivative of distance w.r.t the p-th component of scale
            dp_distance = -(points[row, p] - points[column, p])**2 / \
                (distance * scale[p]**3)

            # derivative of distance w.r.t the p-th component of scale
            if q == p:
                dq_distance = dp_distance
            else:
                dq_distance = -(points[row, q] - points[column, q])**2 / \
                    (distance * scale[q]**3)

            # Second mixed derivative of distance w.r.t the p and q component
            if q == p:
                dpq_distance = ((points[row, p] - points[column, p])**2 /
                                (distance * scale[p]**3)) * \
                        (3.0 / scale[p] + dp_distance / distance)
            else:
                dpq_distance = ((points[row, p] - points[column, p])**2 /
                                (distance**2 * scale[p]**3)) * dq_distance

            # Second partial derivative of correlation w.r.t p and q components
            matrix_data[p, q, index_pointer] = \
                d2_kernel * dp_distance * dq_distance + \
                d1_kernel * dpq_distance

            # Using symmetry of Hessian
            if q != p:
                matrix_data[q, p, index_pointer] = \
                        matrix_data[p, q, index_pointer]
