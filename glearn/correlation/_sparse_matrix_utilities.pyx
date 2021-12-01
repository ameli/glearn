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

import numpy
from ..kernels import Kernel
from ..kernels cimport Kernel

__all__ = ['estimate_kernel_threshold', 'estimate_max_nnz']


# ==============
# gamma function
# ==============

def _gamma_function(dimension):
    """
    Computes the gamma function of the half integer dimension/2+1.

    :param dimension: Dimension of space.
    :type dimension: int

    :return: Gamma function of dimension/2 + 1.
    :rtype: float
    """

    # Compute Gamma(dimension/2 + 1)
    if dimension % 2 == 0:
        k = 0.5 * dimension
        gamma = 1.0
        while k > 0.0:
            gamma *= k
            k -= 1.0
    else:
        k = numpy.ceil(0.5 * dimension)
        gamma = numpy.sqrt(numpy.pi)
        while k > 0.0:
            gamma *= k - 0.5
            k -= 1.0

    return gamma


# ===========
# ball radius
# ===========

def _ball_radius(volume, dimension):
    """
    Computes the radius of n-ball at dimension n, given its volume.

    :param volume: Volume of n-ball
    :type volume: double

    :param dimension: Dimension of embedding space
    :type dimension: int

    :return: radius of n-ball
    :rtype: double
    """

    # Compute gamma function of dimension/2+1
    gamma = _gamma_function(dimension)

    # radius from volume
    radius = (gamma * volume)**(1.0 / dimension) / numpy.sqrt(numpy.pi)

    return radius


# ===========
# ball volume
# ===========

def _ball_volume(radius, dimension):
    """
    Computes the volume of n-ball at dimension n, given its volume.

    :param radius: Volume of n-ball
    :type volume: double

    :param dimension: Dimension of embedding space
    :type dimension: int

    :return: radius of n-ball
    :rtype: double
    """

    # Compute gamma function of dimension/2+1
    gamma = _gamma_function(dimension)

    # volume from radius
    volume = (radius * numpy.sqrt(numpy.pi))**(dimension) / gamma

    return volume


# =========================
# estimate kernel threshold
# =========================

def estimate_kernel_threshold(
        matrix_size,
        dimension,
        density,
        scale,
        kernel):
    """
    Estimates the kernel's tapering threshold to sparsify a dense matrix into a
    sparse matrix with the requested density.

    Here is how density :math:`\\rho` is related to the kernel_threshold
    :math:`\\tau`:

    .. math::

        a = \\rho n = \\mathrm{Vol}_{d}(r/l),
        \\tau = k(r),

    where:

        * :math:`n` is the number of points in the unit hypercube, also it is
          the matrix size.
        * :math:`d` is the dimension of space.
        * :math:`\\mathrm{Vol}_{d}(r/l)` is the volume of d-ball of radius
          :math:`r/l`.
        * :math:`l = 1/(\\sqrt[d]{n} - 1)` is the grid size along each axis,
          assuming the points are places on an equi-distanced structured grid.
        * :math:`k` is the Matern correlation function.
        * :math:`a` is the adjacency of a point, which is the number of
          the neighbor points that are correlated to a point.
        * :math:`\\rho` is the sparse matrix density (input to this function).
        * :math:`\\tau` is the kernel threshold (output of this function).

    The adjacency :math:`a` is the number of points on an integer lattice
    and inside a d-ball. This quantity can be approximated by the volume of a
    d-ball, see for instance
    `Gauss circle problem<https://en.wikipedia.org/wiki/Gauss_circle_problem>`_
     in 2D.

    A non-zero kernel threshold is used to sparsify a matrix by tapering its
    correlation function. However, if the kernel threshold is too large, some
    elements of the correlation matrix will not be correlated to any other
    neighbor point. This leads to a correlation matrix with some rows that have
    only one non-zero element equal to one on the diagonal and zero elsewhere.
    Essentially, if all points loose their correlation to a neighbor, the
    matrix becomes identity.

    This function checks if a set of parameters to form a sparse matrix could
    lead to this issue. The correlation matrix in this module is formed by the
    mutual correlation between spatial set of points in the unit hypercube. We
    assume each point is surrounded by a sphere of the radius of the kernel
    threshold. If this radius is large enough, then the sphere of all points
    intersect. If the criteria is not met, this function raises ``ValueError``.

    :param matrix_size: The size of the square matrix. This is also the number
        of points used to construct the correlation matrix.
    :type matrix_size: int

    :param dimension: The dimension of the space of points used to construct
        the correlation matrix.
    :type dimension: int

    :param sparse_density: The desired density of the sparse matrix. Note that
        the actual density of the generated matrix will not be exactly equal to
        this value. If the matrix size is large, this value is close to the
        actual matrix density.
    :type sparse_density: int

    :param scale: A parameter of correlation function that scales spatial
        distance.
    :type scale: float

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float

    :return: Kernel threshold level
    :rtype: double
    """

    # Number of neighbor points to be correlated in a neighborhood of a point
    adjacency_volume = density * matrix_size

    # If Adjacency is less that one, the correlation matrix becomes identity
    # since no point will be adjacent to other in the correlation matrix.
    if adjacency_volume < 1.0:
        raise ValueError(
                'Adjacency: %0.2f. Correlation matrix will become identity '
                % (adjacency_volume) +
                'since kernel radius is less than grid size. To increase ' +
                'adjacency, consider increasing density or scale.')

    # Volume of an ellipsoid with radii of the components of the correlation
    # scale is equivalent to the volume of an d-ball with the radius of the
    # geometric mean of the correlation scale elements
    dimesnion = scale.size
    geometric_mean_radius = numpy.prod(scale)**(1.0/ dimension)
    correlation_ellipsoid_volume = _ball_volume(geometric_mean_radius,
                                                dimension)

    # Normalize the adjacency volume with the volume of an ellipsoid of the
    # correlation scale radii
    adjacency_volume /= correlation_ellipsoid_volume

    # Approximate radius of n-sphere containing the above number of adjacent
    # points, assuming adjacent points are distanced on integer lattice.
    adjacency_radius = _ball_radius(adjacency_volume, dimension)

    # Number of points along each axis of the grid
    grid_axis_num_points = matrix_size**(1.0 / dimension)

    # Size of grid elements
    grid_size = 1.0 / (grid_axis_num_points - 1.0)

    # Scale the integer lattice of adjacency radius by the grid size.
    # This is the tapering radius of the kernel
    kernel_radius = grid_size * adjacency_radius

    # Threshold of kernel to perform tapering
    kernel_threshold = kernel.kernel(kernel_radius)

    return kernel_threshold


# ================
# estimate max nnz
# ================

def estimate_max_nnz(
        matrix_size,
        scale,
        dimension,
        density):
    """
    Estimates the maximum number of nnz needed to store the indices and data of
    the generated sparse matrix. Before the generation of the sparse matrix,
    its nnz (number of non-zero elements) are not known. Thus, this function
    only guesses this value based on its density.

    :param matrix_size: The size of the square matrix. This is also the number
        of points used to construct the correlation matrix.
    :type matrix_size: int

    :param dimension: The dimension of the space of points used to construct
        the correlation matrix.
    :type dimension: int

    :param sparse_density: The desired density of the sparse matrix. Note that
        the actual density of the generated matrix will not be exactly equal to
        this value. If the matrix size is large, this value is close to the
        actual matrix density.
    :type sparse_density: int

    :return: maximum non-zero elements of sparse array
    :rtype: double
    """

    estimated_nnz = int(numpy.ceil(density * (matrix_size**2)))

    # Normalize correlation scale so that its largest element is one
    normalized_scale = scale / numpy.max(scale)

    # Get the geometric mean of the normalized correlation
    geometric_mean_radius = \
        numpy.prod(normalized_scale)**(1.0/dimension)

    # Multiply the estimated nnz by unit hypercube over unit ball volume ratio
    unit_hypercube_volume = 1.0
    safty_coeff = unit_hypercube_volume / \
        _ball_radius(geometric_mean_radius, dimension)
    max_nnz = int(numpy.ceil(safty_coeff * estimated_nnz))

    return max_nnz
