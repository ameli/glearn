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
from .._utilities.plot_utilities import *                    # noqa: F401, F403
from .._utilities.plot_utilities import load_plot_settings, plt

__all__ = ['generate_data']


# =============
# generate data
# =============

def generate_data(
        points,
        noise_magnitude,
        plot=False):
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

    # If points are 1d array, wrap them to a 2d array
    if points.ndim == 1:
        points = numpy.array([points], dtype=float).T

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
        _plot_data(points, z)

    return z


# =========
# plot data
# =========

def _plot_data(points, z):
    """
    Plots 1D or 2D data.
    """

    load_plot_settings()

    num_points = points.shape[0]
    dimension = points.shape[1]

    if dimension == 1:

        x = points
        xi = numpy.linspace(0, 1)
        zi = generate_data(xi, 0.0, False)

        fig, ax = plt.subplots()
        ax.plot(x, z, 'o', color='black', markersize=4, label='noisy data')
        ax.plot(xi, zi, color='black', label='noise-free data')
        ax.set_xlim([0, 1])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$z(x)$')
        ax.set_title('Sample one dimensional data')
        ax.legend()

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
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(r'$z(x_1, x_2)$')
        ax.set_title('Sample two dimensional data')
        plt.show()

    else:
        raise ValueError('Dimension should be "1" or "2" to plot data.')
