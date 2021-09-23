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
from libc.math cimport sqrt

__all__ = ['euclidean_distance']


# ==================
# euclidean_distance
# ==================

@boundscheck(False)
@wraparound(False)
cdef double euclidean_distance(
        const double[:] point1,
        const double[:] point2,
        const double[:] scale,
        const int dimension) nogil:
    """
    Returns the weighted Euclidean distance between two points.

    :param point1: 1D array of the coordinates of a point
    :type point1: cython memoryview (double)

    :param point2: 1D array of the coordinates of a point
    :type point2: cython memoryview (double)

    :param dimension: Dimension of the coordinates of the points.
    :type dimension: int

    :return: Euclidean distance betwrrn point1 and point2
    :rtype: double
    """

    cdef double distance2 = 0
    cdef int dim

    for dim in range(dimension):
        distance2 += ((point1[dim] - point2[dim]) / scale[dim])**2

    return sqrt(distance2)
