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

__all__ = ['generate_points']


# ===============
# generate points
# ===============

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
