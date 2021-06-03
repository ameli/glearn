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
import matplotlib.pyplot as plt


# =============
# generate data
# =============

def generate_points(num_points, dimension=2, grid=True):
    """
    Generates 2D array of size ``(n, m)`` where ``m`` is the dimension of the
    space, and ``n`` is the number of generated points inside a unit hypercube.
    The number ``n`` is determined as follow:

    * If ``grid`` is ``False``, then ``n`` is ``num_points``.
    * If ``grid`` is ``True``, the ``num_points`` is the number of points
      along each axes of a grid of points. Thus, ``n`` is
      ``num_points**dimension``.

    :param num_points: Number of points of the generated data.
        * If ``grid`` is ``False``, the ``num_points`` is the number of random
          points to be generated in the unit hypecube.
        * If ``grid`` is ``True``, the ``num_points`` is the number of points
          along each axes of a grid of points. Thus, the total number of points
          will be ``num_points**dimension``.
    :param num_points: int

    :param dimension: The dimension of hypercube to which generated points
        inside it.
    :
    type dimension: int
    :param grid: A flag indicating whether the points should be generated on
        a grid of points, or randomly generated.
    :type grid: bool

    :return: 2D array where each row is the coordinate of a point.
    :rtype: numpy.ndarray
    """

    if grid:

        # Grid of points
        axis = numpy.linspace(0, 1, num_points)
        axes = numpy.tile(axis, (dimension, 1))
        mesh = numpy.meshgrid(*axes)

        n = num_points**dimension
        points = numpy.empty((n, dimension), dtype=float)
        for i in range(dimension):
            points[:, i] = mesh[i].ravel()

    else:
        # Randomized points in a square area
        points = numpy.random.rand(num_points, dimension)

    return points


# =============
# generate data
# =============

def generate_data(points, noise_magnitude, plot=False):
    """
    Generates 1D array of data points. The data are the additive ``sin``
    function along each axis plus uniform noise.

    :param points: 2D array of points, where each row represents a point
        coordinate.
    :param points: numpy.ndarray

    :param noise_magntude: The magnitude of additive noise to the data.
    :type noise_magnitude: float

    :return: 1D array of data
    :rtype: numpy.ndarray
    """

    num_points = points.shape[0]
    dimension = points.shape[1]
    z = numpy.zeros((num_points, ), dtype=float)

    for i in range(dimension):
        z += numpy.sin(points[:, i]*numpy.pi)

    # Add noise
    numpy.random.seed(31)
    z += noise_magnitude*numpy.random.randn(num_points)

    # Plot data
    if plot:

        if dimension == 1:

            fig, ax = plt.subplots()
            num_points_on_axis = numpy.sqrt(num_points).astype(int)
            p = ax.plot(points, z)
            plt.show()

        elif dimension == 2:

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            num_points_on_axis = numpy.sqrt(num_points).astype(int)
            x_mesh = points[:, 0].reshape(num_points_on_axis, -1)
            y_mesh = points[:, 1].reshape(num_points_on_axis, -1)
            z_mesh = z.reshape(num_points_on_axis, num_points_on_axis)
            p = ax.plot_surface(x_mesh, y_mesh, z_mesh, linewidth=0,
                                antialiased=False)
            fig.colorbar(p, ax=ax)
            plt.show()

        else:
            raise ValueError('Dimension should be "1" or "2" to plot data.')

    return z


# ========================
# generate basis functions
# ========================

def generate_basis_functions(points, polynomial_degree=2, trigonometric=False):
    """
    Generates basis functions for the mean function of the general linear
    model.
    """

    n = points.shape[0]
    dimension = points.shape[1]

    # Adding polynomial functions
    powers_array = numpy.arange(polynomial_degree + 1)
    # powers_tile = numpy.tile(powers_array, (polynomial_degree+1, 1))
    powers_tile = numpy.tile(powers_array, (dimension, 1))
    powers_mesh = numpy.meshgrid(*powers_tile)

    powers_ravel = []
    for i in range(dimension):
        powers_ravel.append(powers_mesh[i].ravel())

    # The array powers_ravel contains all combinations of powers
    powers_ravel = numpy.array(powers_ravel)

    # For each combination of powers, we compute the power sum
    powers_sum = numpy.sum(powers_ravel, axis=0)

    # The array powers contains those combinations that their sum does not
    # exceed the polynomial_degree
    powers = powers_ravel[:, powers_sum <= polynomial_degree]

    num_degrees = powers.shape[0]
    num_basis = powers.shape[1]

    # Basis functions
    X = numpy.ones((n, num_basis), dtype=float)
    for j in range(num_basis):
        for i in range(num_degrees):
            X[:, j] *= points[:, i]**powers[i, j]

    # Trigonmometric basis functions
    if trigonometric:
        X_trigonometric = numpy.empty((n, 2*dimension))

        for i in range(dimension):
            X_trigonometric[:, i+0] = numpy.sin(points[:, i]*numpy.pi)
            X_trigonometric[:, i+1] = numpy.cos(points[:, i]*numpy.pi)

        # append to the polynomial basis
        X = numpy.c_[X, X_trigonometric]

    return X
