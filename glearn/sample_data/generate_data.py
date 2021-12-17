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
from .._utilities.plot_utilities import load_plot_settings, plt, \
    show_or_save_plot

__all__ = ['generate_data']


# =============
# generate data
# =============

def generate_data(
        points,
        noise_magnitude,
        seed=0,
        plot=False):
    """
    Generates 1D array of data points. The data are the additive ``sin``
    function along each axis plus uniform noise.

    :param points: 2D array of points, where each row represents a point
        coordinate.
    :param points: numpy.ndarray

    :param noise_magnitude: The magnitude of additive noise to the data.
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
    if seed is None:
        rng = numpy.random.RandomState()
    else:
        rng = numpy.random.RandomState(seed)
    z += noise_magnitude*rng.randn(num_points)

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
        ax.set_title('Sample one -dimensional data')
        ax.legend(fontsize='small')

        plt.tight_layout()
        show_or_save_plot(plt, 'data', transparent_background=True)

    elif dimension == 2:

        # Noise free data
        xi = numpy.linspace(0, 1)
        yi = numpy.linspace(0, 1)
        Xi, Yi = numpy.meshgrid(xi, yi)
        XY = numpy.c_[Xi.ravel(), Yi.ravel()]
        zi = generate_data(XY, 0.0, False)
        Zi = numpy.reshape(zi, (xi.size, yi.size))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], z, marker='.', s=7, c='black',
                   label='noisy data')

        surf = ax.plot_surface(Xi, Yi, Zi, linewidth=0, antialiased=False,
                               color='darkgray', label='noise-free data')

        # To avoid a bug in matplotlib
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        x_min = numpy.min(points[:, 0])
        x_max = numpy.max(points[:, 0])
        y_min = numpy.min(points[:, 1])
        y_max = numpy.max(points[:, 1])

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(r'$z(x_1, x_2)$')
        ax.set_title('Sample two-dimensional data')
        ax.legend(fontsize='small')
        ax.view_init(elev=40, azim=120)

        plt.tight_layout()
        show_or_save_plot(plt, 'data', transparent_background=True)

    else:
        raise ValueError('Dimension should be "1" or "2" to plot data.')
