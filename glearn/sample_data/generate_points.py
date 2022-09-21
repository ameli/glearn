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
        dimension=1,
        grid=False,
        a=None,
        b=None,
        contrast=0.0,
        seed=0):
    """
    Generate a set of points in the unit hypercube.

    * Points can be generated either randomly or on a lattice.
    * The density of the distribution of the points can be either uniform over
      all unit hypercube, or can be more concentrated with a higher density in
      a smaller hypercube region embedded inside the unit hypercube.

    Parameters
    ----------

    num_points : int
        Determines the number of generated points as follows:

        * If ``grid`` is `False`, ``num_points`` is the number of random
          points to be generated in the unit hypercube.
        * If ``grid`` is `True`, ``num_points`` is the number of points
          along each axes of a grid of points. Thus, the total number of points
          is ``num_points**dimension``.

    dimension : int, default=1
        The dimension of the space of points.

    grid : bool, default=True
        If `True`, it generates the set of points on a lattice grid. Otherwise,
        it randomly generates points inside the unit hypercube.

    a : float or array_like, default=None
        The coordinate of a corner point of an embedded hypercube inside the
        unit hypercube. The point ``a`` is the closet point of the embedded
        hypercube to the origin. The coordinates of this point should be
        between the origin and the point with coordinates ``(1,1, ..., 1)``. If
        `None`, it is assumed that ``a`` is the origin. When ``dimension`` is
        `1`, ``a`` should be a scalar.

    b : float or array_like, default=None
        The coordinate of another corner point of an embedded hypercube inside
        the unit hypercube. The point ``b`` is the furthest point of the
        embedded hypercube from the origin. The coordinates of this point
        should be between the point ``a`` and the point ``(1,1, ..., 1)``. If
        `None`, it is assumed that the coordinates of this point is all `1`.
        When ``dimension`` is `1``, ``b`` should be a scalar.

    contrast : float, default=0.0
        The extra concentration of points to be generated inside the embedding
        hypercube with the corner points ``a`` and ``b``. Contrast is the
        relative difference of the density of the points inside and outside
        the embedding hypercube region and is between `0` and `1`. When set to
        `0`, all points are generated inside the unit hypercube with uniform
        distribution, hence, there is no difference between the density of the
        points inside and outside the inner hypercube. In contrary, when
        contrast is set to `1`, all points are generated only inside the
        embedding hypercube and no point is generated outside of the inner
        hypercube.

    seed : int, default=0
        Seed number of the random generator, which can be a non-negative
        integer. If set to `None`, the result of the random generator is not
        repeatable.

    Returns
    -------

    x : numpy.ndarray
        A 2D array where each row is the coordinate of a point. The size of the
        array is ``(n, m)`` where ``m`` is the ``dimension`` of the space, and
        ``n`` is the number of generated points inside a unit hypercube. The
        number ``n`` is determined as follow:

        * If ``grid`` is `True`, then ``n = num_points**dimension``.
        * If ``grid`` is `False`, then ``n = num_points``.

    See Also
    --------

    glearn.sample_data.generate_data

    Notes
    -----

    **Grid versus Random Points:**

    Points are generated in a multi-dimensional space inside the unit
    hypercube, either randomly (when ``grid`` is `False`) or on a structured
    grid (if ``grid`` is `True`).

    **Generate Higher Concentration of Points in a Region:**

    The points are generated with uniform distribution inside the unit
    hypercube :math:`[0, 1]^d`. However, it is possible to generate the points
    with higher concentration in a specific hypercube :math:`\\mathcal{H}
    \\subset [0, 1]^d`, which is embedded inside the unit hypercube. The
    coordinates of the inner hypercube :math:`\\mathcal{H}` is determined by
    its two opposite corner points given by the arguments ``a`` and ``b``.

    The argument ``contrast``, here denoted by :math:`\\gamma`, specifies the
    *excessive* relative density of points in :math:`\\mathcal{H}` compared to
    the rest of the unit hypercube. Namely, if :math:`\\rho_{1}` is the density
    of points inside :math:`\\mathcal{H}` and :math:`\\rho_2` is the density of
    points outside :math:`\\mathcal{H}`, then

    .. math::

        \\gamma = \\frac{\\rho_1 - \\rho_2}{\\rho_2}.

    If :math:`\\gamma = 0`, there is no difference between the density of the
    points inside and outside :math:`\\mathcal{H}`, hence all points inside
    :math:`[0, 1]^d` are generated with uniform distribution. If in contrary,
    :math:`\\gamma = 1`, then the density of points outside
    :math:`\\mathcal{H}` is zero and no point is generated outside
    :math:`\\mathcal{H}`. Rather, all points are generated inside this region.

    Examples
    --------

    * Generate 100 random points in the 1-dimensional interval :math:`[0, 1]`:

      .. code-block:: python

          >>> from glearn.sample_data import generate_points
          >>> x = generate_points(100, dimension=1, grid=False)
          >>> x.shape
          (100, 1)

    * Generate 100 random points in the 1-dimensional interval :math:`[0, 1]`
      where :math:`70 \\%` more points are inside :math:`[0.2, 0.5]` and
      :math:`30 \\%` of the points are outside of the latter interval:

      .. code-block:: python

          >>> from glearn.sample_data import generate_points
          >>> x = generate_points(100, dimension=1, grid=False, a=0.2,
          ...                     b=0.5, contrast=0.7)

    * Generate 100 random points on a 2-dimensional square :math:`[0, 1]^2`
      where :math:`70 \\%` more points are inside a rectangle of the points
      :math:`a=(0.2, 0.3)` and :math:`b=(0.4, 0.5)`

      .. code-block:: python

          >>> from glearn.sample_data import generate_points
          >>> x = generate_points(100, dimension=2, grid=False, a=(0.2, 0.3),
          ...                     b=(0.4, 0.5), contrast=0.7)

    * Generate a two-dimensional grid of :math:`30 \\times 30` points in the
      square :math:`[0, 1]^2`:

      .. code-block:: python

          >>> from glearn.sample_data import generate_points
          >>> x = generate_points(30, dimension=2, grid=True)
          >>> x.shape
          (900, 2)
    """

    # Check arguments
    a, b = _check_arguments(num_points, dimension, a, b, contrast, grid)

    if contrast == 0 or a is None or b is None:
        # All points are uniformly distributed in hypercube [0, 1]
        points = _generate_uniform_points(num_points, dimension, grid, seed)

    else:
        # Volume of the inner hypercube
        inner_vol = numpy.prod(b - a)

        # Density of points in the inner interval
        density = ((1.0-inner_vol)*contrast + inner_vol)
        num_inner_points = num_points * density

        # Outer points will be generated in the hypercube [0, 1] which also
        # contains the inner interval. Thus, a fraction of outer points will be
        # count toward inner points. We subtract from the inner points to
        # adjust the density.
        num_inner_points_excess = num_points * (1.0-density) * \
            (inner_vol) / (1.0 - inner_vol)
        num_inner_points = num_inner_points - num_inner_points_excess

        # Convert to integer
        num_inner_points = int(num_inner_points + 0.5)
        if num_inner_points > num_points:
            num_inner_points = num_points

        # The remain should be the outer points which are generated in both
        # inner and outer regions (the whole unit hypercube) with uniform
        # distribution.
        num_outer_points = num_points - num_inner_points

        # Generate inner and outer points with uniform distributions
        inner_points = _generate_uniform_points(num_inner_points, dimension,
                                                grid, seed)
        outer_points = _generate_uniform_points(num_outer_points, dimension,
                                                grid, seed)

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

def _check_arguments(num_points, dimension, a, b, contrast, grid):
    """
    Checks the arguments.
    """

    # Check num_points
    if not isinstance(num_points, (int, numpy.int64)):
        raise TypeError('"num_points" should be an integer.')
    elif num_points < 2:
        raise ValueError('"num_points" should be greater than 1.')

    # Check dimension
    if not isinstance(dimension, (int, numpy.int64)):
        raise TypeError('"dimension" should be an integer.')
    elif dimension < 1:
        raise ValueError('"dimension" should be greater or equal to 1.')

    # Check a
    if a is not None:
        if not isinstance(
                a, (int, numpy.int64, float, numpy.ndarray, list, tuple)):
            raise TypeError('"a" should be a real number, list, or an array.')
        if isinstance(a, (list, tuple)):
            a = numpy.array(a, dtype=float)
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
        if not isinstance(
                a, (int, numpy.int64, float, numpy.ndarray, list, tuple)):
            raise TypeError('"a" should be a real number, list, or an array.')
        if isinstance(b, (list, tuple)):
            b = numpy.array(b, dtype=float)
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

    # Check contrast
    if not isinstance(contrast, (int, numpy.int64, float)):
        raise TypeError('"contrast" should be a real number.')
    elif a is None and contrast != 0.0:
        raise ValueError('"contrast" should be zero when "a" and "b" are ' +
                         'None.')
    elif contrast < 0.0 or contrast > 1.0:
        raise ValueError('"contrast" should be between or equal to 0.0 and ' +
                         '1.0.')

    # Check grid
    if not isinstance(grid, bool):
        raise TypeError('"grid" should be boolean.')

    return a, b


# =======================
# generate uniform points
# =======================

def _generate_uniform_points(num_points, dimension, grid, seed):
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
        if seed is None:
            rng = numpy.random.RandomState()
        else:
            rng = numpy.random.RandomState(seed)
        points = rng.rand(num_points, dimension)

    return points
