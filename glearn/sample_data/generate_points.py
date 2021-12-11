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

def generate_points(
        num_points,
        dimension=2,
        grid=False,
        a=None,
        b=None,
        ratio=0.0):
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
          points to be generated in the unit hypercube.
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

    # Check arguments
    _check_arguments(num_points, dimension, a, b, ratio, grid)

    if ratio == 0 or a is None or b is None:
        # All points are uniformly distributed in hypercube [0, 1]
        points = _generate_uniform_points(num_points, dimension, grid)

    else:
        # Inner volume
        inner_volume = numpy.prod(b - a)

        # Outer points will be generated in the hypercube [0, 1] which also
        # contains the inner interval. Thus, a fraction of outer points will be
        # counter toward inner points. To compensate the excess outer points,
        # we adjust the ratio based on the volume of the inner interval.
        ratio = (ratio - inner_volume) / (1.0 - inner_volume)

        # Points inside and outside the sub-interval
        num_inner_points = int(ratio * num_points + 0.5)
        num_outer_points = num_points - num_inner_points

        # Generate inner points
        inner_points = _generate_uniform_points(
                num_inner_points, dimension, grid)
        outer_points = _generate_uniform_points(
                num_outer_points, dimension, grid)

        if dimension == 1:
            a = numpy.asarray([a])
            b = numpy.asarray([b])

        # Translate inner points to the sub-interval
        for i in range(dimension):
            inner_points[:, i] = a[i] + inner_points[:, i] * (b[i]-a[i])

        # Merge inner and outer points
        points = numpy.r_[inner_points, outer_points]

        # Sort points
        if dimension == 1:
            points = numpy.sort(points)

    return points


# ===============
# check arguments
# ===============

def _check_arguments(num_points, dimension, a, b, ratio, grid):
    """
    Checks the arguments.
    """

    # Check num_points
    if not isinstance(num_points, int):
        raise TypeError('"num_points" should be an integer.')
    elif num_points < 2:
        raise ValueError('"num_points" should be greater than 1.')

    # Check dimension
    if not isinstance(dimension, int):
        raise TypeError('"dimension" should be an integer.')
    elif dimension < 1:
        raise ValueError('"dimension" should be greater or equal to 1.')

    # Check a
    if a is not None:
        if not isinstance(a, (int, float, numpy.ndarray)):
            raise TypeError('"a" should be a real number or an array.')
        if numpy.isscalar(a) and dimension != 1:
            raise ValueError('"a" should be a scalar when "dimension" is 1.')
        elif isinstance(a, numpy.ndarray) and a.size != dimension:
            raise ValueError('"a" size should be equal to "dimension".')
        elif numpy.any(a < 0.0):
            raise ValueError('All components of "a" should be greater or ' +
                             'equal to 0.0.')
    else:
        if b is not None:
            raise ValueError('"a" and "b" should be both None or not None.')

    # Check b
    if b is not None:
        if not isinstance(a, (int, float, numpy.ndarray)):
            raise TypeError('"a" should be a real number or an array.')
        if numpy.isscalar(b) and dimension != 1:
            raise ValueError('"b" should be a scalar when "dimension" is 1.')
        elif isinstance(b, numpy.ndarray) and b.size != dimension:
            raise ValueError('"b" size should be equal to "dimension".')
        elif numpy.any(b > 1.0):
            raise ValueError('All component of "b" should be less or equal ' +
                             'to 1.0.')
        elif numpy.any(b <= a):
            raise ValueError('All component of "b" should be greater than ' +
                             '"a".')

    # Check ratio
    if not isinstance(ratio, (int, float)):
        raise TypeError('"ratio" should be a real number.')
    elif a is None and ratio != 0.0:
        raise ValueError('"ratio" should be zero when "a" and "b" are None.')
    elif ratio < 0.0 or ratio > 1.0:
        raise ValueError('"ratio" should be between or equal to 0.0 and 1.0.')

    # Check grid
    if not isinstance(grid, bool):
        raise TypeError('"grid" should be boolean.')


# =======================
# generate uniform points
# =======================

def _generate_uniform_points(num_points, dimension, grid):
    """
    Generates points in the hypercube [0, 1]^dimension using uniform
    distribution.
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
