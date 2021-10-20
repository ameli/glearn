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
import scipy
from scipy import sparse
import multiprocessing

# Cython
from cython.parallel cimport parallel, prange
from libc.stdio cimport printf
from libc.stdlib cimport exit, malloc, free
from ._compute_sparse_correlation cimport compute_sparse_correlation
from ._sparse_matrix_utilities import estimate_kernel_threshold, \
        estimate_max_nnz
from ..kernels import Kernel
from ..kernels cimport Kernel
cimport cython
cimport openmp

__all__ = ['sparse_cross_correlation']


# ===========================
# generate correlation matrix
# ===========================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _generate_correlation_matrix(
        const double[:, ::1] training_points,
        const double[:, ::1] test_points,
        const int num_training_points,
        const int num_test_points,
        const int dimension,
        const double[:] scale,
        Kernel kernel,
        const double kernel_threshold,
        const int num_threads,
        const long max_nnz,
        long[:] nnz,
        long[:] matrix_row_indices,
        long[:] matrix_column_indices,
        double[:] matrix_data) nogil:
    """
    Generates a sparse correlation matrix.

    In this function, we pre-allocated array of sparse matrix with presumed nnz
    equal to ``max_nnz``. If the number of required nnz is more than that, this
    function should be stopped and a newer array with larger memory should be
    pre-allocated. To stop openmp loop, we cannot use ``break``. Instead we use
    the ``success`` variable. A zero success variable signals other threads to
    perform idle loops till end. This way, the openmp is terminated gracefully.

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

    :param scale: A parameter of the correlation function that scales spatial
        distances.
    :type scale: double

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float

    :param kernel_threshold: The parameter tapers the correlation kernel. The
        kernel values below kernel threshold are assumed to be zero, which
        sparsifies the matrix.
    :type kernel_threshold: double

    :param num_threads: Number of parallel threads in openmp.
    :type num_threads: int

    :param max_nnz: The size of pre-allocated sparse matrix. The generated
        sparse matrix many have less or more nnz than this value. If the matrix
        requires more nnz, max_nnz should be increased accordingly and this
        function should be recalled.
    :type max_nnz: long

    :param nnz: An output variable, showing the actual nnz of the generated
        sparse matrix.
    :type nnz: long

    :param matrix_row_indices: The row indices of sparse matrix in COO format.
        The size of this array is equal to max_nnz. However, in practice, the
        matrix may have smaller nnz.
    :type matrix_row_indices: cython memoryview (long)

    :param matrix_column_indices: The column indices of sparse matrix in COO
        format. The size of this array is equal to max_nnz. However, in
        practice, the matrix may have smaller nnz.
    :type matrix_column_indices: cython memoryview (long)

    :param matrix_data: The non-zero data of sparse matrix in COO format. The
        size of this array is max_nnz.
    :type matrix_data: cython memoryview (double)

    :return: success of the process. If the required nnz to generate the sparse
        matrix needs to be larger than the preassigned max_nnz, this function
        is terminated and ``0`` is returned. However, if the required nnz is
        less than or equal to max_nnz and the sparse matrix is generated
        successfully, ``1`` (success) is returned.
    :rtype: int
    """

    cdef long i, j
    cdef int dim
    cdef double *thread_data = <double *> malloc(num_threads * sizeof(double))
    cdef int[1] success

    # Set number of parallel threads
    openmp.omp_set_num_threads(num_threads)

    # Initialize openmp lock to setup a critical section
    cdef openmp.omp_lock_t lock
    openmp.omp_init_lock(&lock)

    # Using max possible chunk size for parallel threads
    cdef int chunk_size = int((<double> num_training_points) / num_threads)
    if chunk_size < 1:
        chunk_size = 1

    # Iterate over rows of matrix
    nnz[0] = 0
    success[0] = 1
    with nogil, parallel():
        for i in prange(num_training_points, schedule='dynamic',
                        chunksize=chunk_size):

            # If one of threads was not successful, do empty loops till end
            # we cannot break openmp loop, so here we just continue
            if not success[0]:
                continue

            for j in range(num_test_points):

                # Compute an element of the matrix
                thread_data[openmp.omp_get_thread_num()] = \
                        compute_sparse_correlation(
                                training_points[i][:], test_points[j][:],
                                dimension, scale, kernel)

                # Check with kernel threshold to taper out or store
                if thread_data[openmp.omp_get_thread_num()] > kernel_threshold:

                    # Halt operation if preallocated sparse array is not enough
                    if nnz[0] >= max_nnz:

                        # Critical section
                        openmp.omp_set_lock(&lock)

                        # Avoid duplicate if success is falsified already
                        if success[0]:
                            printf('Pre-allocated sparse array reached max ')
                            printf('nnz: %ld. Terminate operation.\n', max_nnz)
                            success[0] = 0

                        openmp.omp_unset_lock(&lock)

                        # The inner loop is not an openmp loop, so we can break
                        break

                    # Add data to the arrays in an openmp critical section
                    openmp.omp_set_lock(&lock)

                    nnz[0] += 1
                    matrix_row_indices[nnz[0]-1] = i
                    matrix_column_indices[nnz[0]-1] = j
                    matrix_data[nnz[0]-1] = \
                        thread_data[openmp.omp_get_thread_num()]

                    # Release lock to end the openmp critical section
                    openmp.omp_unset_lock(&lock)

    free(thread_data)

    return success[0]


# ========================
# sparse cross correlation
# ========================

def sparse_cross_correlation(
        training_points,
        test_points,
        scale,
        kernel,
        density,
        verbose=False):
    """
    Generates either a sparse correlation matrix in CSR format.

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

    :param scale: A parameter of correlation function that scales spatial
        distance.
    :type scale: float

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float

    :param density: The desired density of the non-zero elements of the sparse
        matrix. Note that the actual density of the generated matrix may not be
        exactly equal to this value. If the matrix size is large, this value is
        close to the actual matrix density.
    :type sparse_density: int

    :param verbose: If ``True``, prints some information during the process.
    :type verbose: bool

    :return: Correlation matrix. If ``points`` is ``n*m`` array, the
        correlation ``K`` is ``n*n`` matrix.
    :rtype: numpy.ndarray or scipy.sparse array

    .. warning::
        ``kernel_threshold`` should be large enough so that the correlation
        matrix is not shrunk to identity. If such case happens, this function
        raises a *ValueError* exception.

        In addition, ``kernel_threshold`` should be small enough to not
        eradicate its positive-definiteness. This is not checked by this
        function and the user should be aware of it.
    """

    # Size of cross correlation matrix
    num_training_points = training_points.shape[0]
    num_test_points = test_points.shape[0]
    matrix_size = numpy.max([num_training_points, num_test_points])
    dimension = training_points.shape[1]

    # Get number of CPU threads
    num_threads = multiprocessing.cpu_count()

    # kernel threshold
    kernel_threshold = estimate_kernel_threshold(
            matrix_size, dimension, density, scale, kernel)

    # maximum nnz
    max_nnz = estimate_max_nnz(matrix_size, scale, dimension, density)

    success = 0

    # Try with the estimated nnz. If not enough, we will double and retry
    while not bool(success):

        # Allocate sparse arrays
        matrix_row_indices = numpy.zeros((max_nnz,), dtype=int)
        matrix_column_indices = numpy.zeros((max_nnz,), dtype=int)
        matrix_data = numpy.zeros((max_nnz,), dtype=float)
        nnz = numpy.zeros((1,), dtype=int)

        # Generate matrix assuming the estimated nnz is enough
        success = _generate_correlation_matrix(
                training_points, test_points, num_training_points,
                num_test_points, dimension, scale, kernel, kernel_threshold,
                num_threads, max_nnz, nnz, matrix_row_indices,
                matrix_column_indices, matrix_data)

        # Double the number of pre-allocated nnz and try again
        if not bool(success):
            max_nnz = 2 * max_nnz
            print('Retry generation of sparse matrix with max_nnz: %d'
                  % (max_nnz))

    # Construct scipy.sparse.coo_matrix, then convert it to CSR matrix.
    correlation_matrix = scipy.sparse.coo_matrix(
            (matrix_data[:nnz[0]],
             (matrix_row_indices[:nnz[0]],
              matrix_column_indices[:nnz[0]])),
            shape=(num_training_points, num_test_points)).tocsr()

    # Actual sparse density
    if verbose:
        actual_density = correlation_matrix.nnz / \
                numpy.prod(correlation_matrix.shape)
        print('Generated sparse correlation matrix using ' +
              'kernel threshold: %0.4f and ' % (kernel_threshold) +
              'sparse density: %0.2e.' % (actual_density))

    return correlation_matrix
