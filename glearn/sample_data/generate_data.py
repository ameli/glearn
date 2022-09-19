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
    Generate noisy sinusoidal data on a set of points.

    Parameters
    ----------

    points : numpy.ndarray
        2D array of size :math:`(n, d)` representing :math:`n` points where
        each row represents an :math:`d`-dimensional coordinate of a point.

    noise_magnitude : float
        The magnitude of additive noise to the data.

    seed : int, default=0
        Seed of the random generator which can be a non-negative integer. If
        set to `None`, the result of the random generator is not repeatable.

    plot : bool, default=False
        If `True`, the data will be plotted (only if the data is 1D or 2D).
        If no display is available (such as executing on remote machines) the
        plot is saved in the current directory in `SVG` format.

    Returns
    -------

    data : numpy.array
        1D array of data of the size :math:`n`.

    See Also
    --------

    glearn.sample_data.generate_points

    Notes
    -----

    Given a set of points :math:`\\{ \\boldsymbol{x}_i \\}_{i = 1}^n` in
    :math:`\\mathbb{R}^d` each with coordinates
    :math:`\\boldsymbol{x}_i = (x_i^1, \\dots, x_i^d)`, this function generates
    the data :math:`y_i = f(\\boldsymbol{x}_i)` where

    .. math::

        y_i = \\sum_{j=1}^d \\sin(x_i^j \\pi) + e,

    where :math:`e \\sim \\mathcal{N}(0, \\epsilon)` is an additive noise with
    normal distribution and noise magnitude :math:`\\epsilon`.

    **Plotting:**

    If ``plot`` is set to `True`, it plots the data.

    * If no graphical backend exists (such as running the code on a remote
      server or manually disabling the X11 backend), the plot will not be
      shown, rather, it will be saved as an ``svg`` file in the current
      directory.
    * If the executable ``latex`` is available on ``PATH``, the plot is
      rendered using :math:`\\rm\\LaTeX` and it may take slightly longer to
      produce the plot.
    * If :math:`\\rm\\LaTeX` is not installed, it uses any available San-Serif
      font to render the plot.

    To manually disable interactive plot display and save the plot as
    ``svg`` instead, add the following at the very beginning of your code
    before importing :mod:`imate`:

    .. code-block:: python

        >>> import os
        >>> os.environ['GLEARN_NO_DISPLAY'] = 'True'

    Examples
    --------

    **One-dimensional Data:**

    Generate 100 random points in a 1-dimensional interval :math:`[0, 1]`
    where :math:`80 \\%` more points are inside :math:`[0.2, 0.4]` compared to
    the outside of the latter interval. Then, generate a sinusoidal function
    with noise magnitude :math:`0.1` on the points.

    .. code-block:: python

        >>> from glearn.sample_data import generate_points, generate_data
        >>> points = generate_points(100, grid=False, a=0.2, b=0.4,
        ...                          contrast=0.8)

        >>> # Generate sample data
        >>> data = generate_data(points, noise_magnitude=0.1, seed=0,
        ...                      plot=True)

    .. image:: ../_static/images/plots/generate_data_1d.png
        :align: center
        :width: 65%
        :class: custom-dark

    **Two-dimensional Data:**

    Generate 100 random points on a 2-dimensional square :math:`[0, 1]^2`
    where :math:`70 \\%` more points are inside a rectangle with the corner
    points :math:`a=(0.2, 0.3)` and :math:`b=(0.4, 0.5)`. Then, generate a
    noisy sinusoidal function on the set of points.

    .. code-block:: python

        >>> from glearn.sample_data import generate_points, generate_data
        >>> points = generate_points(100, dimension=2, grid=False,
        ...                          a=(0.2, 0.3), b=(0.4, 0.5), contrast=0.7)

        >>> # Generate sample data
        >>> data = generate_data(points, noise_magnitude=0.1, seed=0,
        ...                      plot=True)

    .. image:: ../_static/images/plots/generate_data_2d.png
        :align: center
        :width: 85%
        :class: custom-dark
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
        ax.set_ylabel(r'$y(x)$')
        ax.set_title('Sample one-dimensional data')
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
        ax.set_zlabel(r'$y(x_1, x_2)$')
        ax.set_title('Sample two-dimensional data')
        ax.legend(fontsize='small')
        ax.view_init(elev=40, azim=120)

        plt.tight_layout()
        show_or_save_plot(plt, 'data', transparent_background=True)

    else:
        raise ValueError('Dimension should be "1" or "2" to plot data.')
