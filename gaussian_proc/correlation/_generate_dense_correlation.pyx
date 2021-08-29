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

# Python
import numpy
import multiprocessing

# Cython
from cython.parallel cimport parallel, prange
from libc.stdlib cimport exit, malloc, free
from ._compute_dense_correlation cimport compute_dense_correlation, \
        compute_dense_correlation_jacobian, compute_dense_correlation_hessian
from ..kernels import Kernel
from ..kernels cimport Kernel
cimport cython
cimport openmp

__all__ = ['generate_dense_correlation']


# ===========================
# generate correlation matrix
# ===========================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _generate_correlation_matrix(
        const double[:, ::1] points,
        const int matrix_size,
        const int dimension,
        const double[:] distance_scale,
        Kernel kernel,
        const int num_threads,
        double[:, ::1] correlation_matrix) nogil:
    """
    Generates a dense correlation matrix.

    :param points: A 2D array containing the coordinates of the spatial set of
        points in the unit hypercube. The first index of this array is the
        point ids and the second index is the dimension of the coordinates.
    :type points: cython memoryview (double)

    :param matrix_size: The shape of the first index of ``points``, which is
        also the size of the generated output matrix.
    :type matrix_size: int

    :param dimension: The shape of the second index of ``points`` array, which
        is the dimension of the spatial points.
    :type dimension: int

    :param distance_scale: A parameter of the correlation function that
        scales distances.
    :type distance_scale: double

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float

    :param num_threads: Number of parallel threads in openmp.
    :type num_threads: int

    :param correlation_matrix: Output array. Correlation matrix of the size
        ``matrix_size``.
    :type correlation_matrix: cython memoryview (double)
    """

    cdef int i, j
    cdef int dim

    # Set number of parallel threads
    openmp.omp_set_num_threads(num_threads)

    # Using max possible chunk size for parallel threads
    cdef int chunk_size = int((<double> matrix_size) / num_threads)
    if chunk_size < 1:
        chunk_size = 1

    # Iterate over rows of correlation matrix
    with nogil, parallel():
        for i in prange(matrix_size, schedule='static', chunksize=chunk_size):
            for j in range(i, matrix_size):

                # Compute correlation
                correlation_matrix[i][j] = compute_dense_correlation(
                        points[i][:],
                        points[j][:],
                        dimension,
                        distance_scale,
                        kernel)

                # Use symmetry of the correlation matrix
                if i != j:
                    correlation_matrix[j][i] = correlation_matrix[i][j]


# ====================================
# generate correlation matrix jacobian
# ====================================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _generate_correlation_matrix_jacobian(
        const double[:, ::1] points,
        const int matrix_size,
        const int dimension,
        const double[:] distance_scale,
        Kernel kernel,
        const int num_threads,
        double[:, :, ::1] correlation_matrix_jacobian) nogil:
    """
    Generates a dense correlation matrix.

    :param points: A 2D array containing the coordinates of the spatial set of
        points in the unit hypercube. The first index of this array is the
        point ids and the second index is the dimension of the coordinates.
    :type points: cython memoryview (double)

    :param matrix_size: The shape of the first index of ``points``, which is
        also the size of the generated output matrix.
    :type matrix_size: int

    :param dimension: The shape of the second index of ``points`` array, which
        is the dimension of the spatial points.
    :type dimension: int

    :param distance_scale: A parameter of the correlation function that
        scales distances.
    :type distance_scale: double

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float

    :param num_threads: Number of parallel threads in openmp.
    :type num_threads: int

    :param correlation_matrix: Output array. Correlation matrix of the size
        ``matrix_size``.
    :type correlation_matrix: cython memoryview (double)
    """

    cdef int i, j, p
    cdef int dim

    # Set number of parallel threads
    openmp.omp_set_num_threads(num_threads)

    # Using max possible chunk size for parallel threads
    cdef int chunk_size = int((<double> matrix_size) / num_threads)
    if chunk_size < 1:
        chunk_size = 1

    # Iterate over rows of correlation matrix
    with nogil, parallel():
        for i in prange(matrix_size, schedule='static', chunksize=chunk_size):
            for j in range(i, matrix_size):

                compute_dense_correlation_jacobian(
                        points,
                        dimension,
                        distance_scale,
                        kernel,
                        correlation_matrix_jacobian,
                        i, j)

                # Use anti-symmetry of the correlation matrix jacobian
                if i != j:
                    for p in range(dimension):
                        correlation_matrix_jacobian[p][j][i] = \
                            -correlation_matrix_jacobian[p][i][j]


# ===================================
# generate correlation matrix hessian
# ===================================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _generate_correlation_matrix_hessian(
        const double[:, ::1] points,
        const int matrix_size,
        const int dimension,
        const double[:] distance_scale,
        Kernel kernel,
        const int num_threads,
        double[:, :, :, ::1] correlation_matrix_hessian) nogil:
    """
    Generates a dense correlation matrix.

    :param points: A 2D array containing the coordinates of the spatial set of
        points in the unit hypercube. The first index of this array is the
        point ids and the second index is the dimension of the coordinates.
    :type points: cython memoryview (double)

    :param matrix_size: The shape of the first index of ``points``, which is
        also the size of the generated output matrix.
    :type matrix_size: int

    :param dimension: The shape of the second index of ``points`` array, which
        is the dimension of the spatial points.
    :type dimension: int

    :param distance_scale: A parameter of the correlation function that
        scales distances.
    :type distance_scale: double

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float

    :param num_threads: Number of parallel threads in openmp.
    :type num_threads: int

    :param correlation_matrix: Output array. Correlation matrix of the size
        ``matrix_size``.
    :type correlation_matrix: cython memoryview (double)
    """

    cdef int i, j, p, q
    cdef int dim

    # Set number of parallel threads
    openmp.omp_set_num_threads(num_threads)

    # Using max possible chunk size for parallel threads
    cdef int chunk_size = int((<double> matrix_size) / num_threads)
    if chunk_size < 1:
        chunk_size = 1

    # Iterate over rows of correlation matrix
    with nogil, parallel():
        for i in prange(matrix_size, schedule='static', chunksize=chunk_size):
            for j in range(i, matrix_size):

                compute_dense_correlation_hessian(
                        points,
                        dimension,
                        distance_scale,
                        kernel,
                        correlation_matrix_hessian,
                        i, j)

                # Use anti-symmetry of the correlation matrix jacobian
                if i != j:
                    for p in range(dimension):
                        for q in range(dimension):
                            correlation_matrix_hessian[q][p][j][i] = \
                                correlation_matrix_hessian[q][p][i][j]


# ==========================
# generate dense correlation
# ==========================

def generate_dense_correlation(
        points,
        distance_scale,
        kernel,
        derivative,
        verbose):
    """
    Generates a dense correlation matrix.

    .. note::

        If the ``kernel_threshold`` is large, it causes:

            * The correlation matrix :math:`\\mathbf{K}` will not be
              positive-definite.
            * The function :math:`\\mathrm{trace}\\left((\\mathbf{K}+t
              \\mathbf{I})^{-1}\\right)` produces unwanted oscillations.

    :param points: 2D array of the coordinates of the set of points. The first
        index of the array is the point Ids and its size determines the size of
        the correlation matrix. The second index of the array corresponds to
        the dimension of the spatial points.
    :type points: numpy.ndarray

    :param distance_scale: A parameter of correlation function that scales
        distance.
    :type distance_scale: float

    :param verbose: If ``True``, prints some information during the process.
    :type verbose: bool

    :return: Correlation matrix. If ``points`` is ``n*m`` array, the
        correlation matrix has ``n*n`` shape.
    :rtype: numpy.ndarray

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float
    """

    # size of data and the correlation matrix
    matrix_size = points.shape[0]
    dimension = points.shape[1]

    if verbose:
        print('Generated dense correlation matirx of size: %d.'
              % (matrix_size))

    # Get number of CPU threads
    num_threads = multiprocessing.cpu_count()

    if derivative == 0:

        # Initialize matrix
        correlation_matrix = numpy.zeros(
                (matrix_size, matrix_size),
                dtype=float)

        # Dense correlation matrix
        _generate_correlation_matrix(
                points,
                matrix_size,
                dimension,
                distance_scale,
                kernel,
                num_threads,
                correlation_matrix)

        return correlation_matrix

    elif derivative == 1:

        # Initialize matrix
        correlation_matrix_jacobian = numpy.zeros(
                (dimension, matrix_size, matrix_size),
                dtype=float)

        # Dense correlation matrix
        _generate_correlation_matrix_jacobian(
                points,
                matrix_size,
                dimension,
                distance_scale,
                kernel,
                num_threads,
                correlation_matrix_jacobian)

        return correlation_matrix_jacobian

    elif derivative == 2:

        # Initialize matrix
        correlation_matrix_hessian = numpy.zeros(
                (dimension, dimension, matrix_size, matrix_size),
                dtype=float)

        # Dense correlation matrix
        _generate_correlation_matrix_hessian(
                points,
                matrix_size,
                dimension,
                distance_scale,
                kernel,
                num_threads,
                correlation_matrix_hessian)

        return correlation_matrix_hessian

    else:
        raise NotImplementedError('"derivative" can only be "0", "1", or "2".')
