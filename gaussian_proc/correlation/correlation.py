# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# ======
# Import
# ======

import numpy
from ._generate_dense_correlation import generate_dense_correlation
from ._generate_sparse_correlation import generate_sparse_correlation
from ..kernels import Kernel, Matern

try:
    from .._utilities.plot_utilities import matplotlib, plt
    from .._utilities.plot_utilities import load_plot_settings, save_plot
    plot_modules_exist = True
except ImportError:
    plot_modules_exist = False

__all__ = ['Correlation']


# ===========
# Correlation
# ===========

class Correlation(object):
    """
    """

    def __init__(
            self,
            points,
            kernel=None,
            distance_scale=None,
            sparse=False,
            density=1e-3):
        """
        """

        # Check points
        if points is None:
            raise ValueError('"points" cannot be None.')
        elif not isinstance(points, numpy.ndarray):
            raise TypeError('"points" should be a type of "numpy.ndarray".')
        elif points.ndim != 1 and points.ndim != 2:
            raise ValueError('"points" should be either a column vector or ' +
                             'a 2D matrix.')
        elif points.shape[0] < 2:
            raise ValueError('"points" array should contain at least two ' +
                             'points.')

        # set kernel
        if kernel is not None:
            if not isinstance(kernel, Kernel):
                raise TypeError('"kernel" should be an object of "Kernel" ' +
                                'subclasses.')

        # Use default kernel if no kernel is given
        if kernel is None:
            self.kernel = Matern()
        else:
            self.kernel = kernel

        # Attributes
        self.points = points
        self.sparse = sparse
        self.density = density
        self.distance_scale = self.set_distance_scale(distance_scale)

        # Initialize correlation matrix
        self.K_der0 = None
        self.K_der1 = None
        self.K_der2 = None

        # Keeps which of the derivatives are updated (used only for sparse)
        self.K_der0_updated = False
        self.K_der1_updated = False
        self.K_der2_updated = False

    # ==================
    # set distance scale
    # ==================

    def set_distance_scale(self, distance_scale):
        """
        """

        # Check correlation scale
        if distance_scale is None:

            # distance_scale will be either set later, or will be found as
            # additional optimization variable
            distance_scale_ = None

        else:

            if numpy.isscalar(distance_scale):
                if not isinstance(distance_scale, (int, numpy.integer)) and \
                   not isinstance(distance_scale, float):
                    raise TypeError('"distance_scale" should be float.')

                # Convert distance scale to numpy array
                distance_scale_ = numpy.array([distance_scale], dtype=float)

            elif isinstance(distance_scale, list):
                distance_scale_ = numpy.array(distance_scale)

            elif not isinstance(distance_scale, numpy.ndarray):
                raise TypeError('"distance_scale" should be either a scalar,' +
                                ' a list of numbers, or a numpy array.')

            else:
                distance_scale_ = distance_scale

            # if distance_scale is an array of length one, extend the array to
            # be the size of dimension
            dimension = self.points.shape[1]
            if distance_scale_.size == 1:

                # Repeat correlation scale to an array of size dimension
                distance_scale_ = numpy.repeat(distance_scale, dimension)

            elif distance_scale_.size != dimension:
                # Check dimension matches the size of distance_scale array
                raise ValueError('"distance_scale" should have the same ' +
                                 'dimension as the "points".')

        return distance_scale_

    # ==========
    # get matrix
    # ==========

    def get_matrix(
            self,
            distance_scale=None,
            derivative=0,
            plot=False,
            verbose=False):
        """
        Returns the correlation matrix. If the correlation is not available as
        a matrix, it generates the matrix from the kernel and spatial distance
        of the given points.
        """

        # distance scale (if None, uses the distance_scale that this class was
        # initialized with).
        if distance_scale is not None:
            distance_scale_ = self.set_distance_scale(distance_scale)
        else:
            distance_scale_ = self.distance_scale

        if distance_scale_ is None:
            raise ValueError('"distance_scale" cannot be None.')

        # Check arguments
        if derivative not in (0, 1, 2):
            raise ValueError('"derivative" should be 0, 1, or 2.')

        # Initialize variable to determine whether to regenerate matrix or not.
        generate_matrix = False
        correlation_scale_changed = False

        # Note, these if conditions are independent
        if (derivative == 0) and (self.K_der0 is None):
            generate_matrix = True
        if (derivative == 1) and (self.K_der1 is None):
            generate_matrix = True
        if (derivative == 2) and (self.K_der2 is None):
            generate_matrix = True

        if any(self.distance_scale != distance_scale_):
            generate_matrix = True
            correlation_scale_changed = True

        if (derivative == 0) and (not self.K_der0_updated):
            generate_matrix = True
        if (derivative == 1) and (not self.K_der1_updated):
            generate_matrix = True
        if (derivative == 2) and (not self.K_der2_updated):
            generate_matrix = True

        if generate_matrix:

            # Sparse matrix of derivative 1 and 2 needs matrix of derivative 0
            if (derivative > 0) and self.sparse and (self.K_der0 is None):
                # Before generating matrix of derivative 1 or 2, first,
                # generate correlation matrix of derivative 0
                temp_derivative = 0
                self._generate_correlation_matrix(
                        distance_scale_, temp_derivative, self.sparse,
                        self.density, plot, verbose)

            self._generate_correlation_matrix(
                    distance_scale_, derivative, self.sparse, self.density,
                    plot, verbose)

            # Update distance_scale
            self.distance_scale = distance_scale_

        if derivative == 0:
            self.K_der0_updated = True

            if correlation_scale_changed:
                self.K_der1_updated = False
                self.K_der2_updated = False

            return self.K_der0

        elif derivative == 1:
            self.K_der1_updated = True

            if correlation_scale_changed:
                self.K_der0_updated = False
                self.K_der2_updated = False

            return self.K_der1

        elif derivative == 2:
            self.K_der2_updated = True

            if correlation_scale_changed:
                self.K_der0_updated = False
                self.K_der1_updated = False

            return self.K_der2

    # ===========================
    # generate correlation matrix
    # ===========================

    def _generate_correlation_matrix(
            self,
            distance_scale,
            derivative,
            sparse,
            density,
            plot,
            verbose):
        """
        Generates symmetric and positive-definite matrix.

        **Correlation Function:**

        The generated matrix is a correlation matrix based on Matern
        correlation of spatial distance of a list of points in the unit
        hypercube. The Matern correlation function accepts the correlation
        scale parameter :math:`\\rho \\in (0,1]`. Smaller decorrelation
        produces correlation matrix that is closer to the identity matrix.

        **Sparsification:**

        The values of the correlation matrix are between :math:`0` and
        :math:`1`. To sparsify the matrix, the correlation kernel below a
        certain threshold value is set to zero to which tapers the correlation
        kernel. Such threshold can be set through the parameter ``density``,
        which sets an approximate density of the non-zero elements of the
        sparse matrix.

        .. note::

            Setting a too small ``density`` might eradicate the
            positive-definiteness of the correlation matrix.

        **Plotting:**

        If the option ``plot`` is set to ``True``, it plots the generated
        matrix.

        * If no graphical backend exists (such as running the code on a remote
          server or manually disabling the X11 backend), the plot will not be
          shown, rather, it will be saved as an ``svg`` file in the current
          directory.
        * If the executable ``latex`` is on the path, the plot is rendered
          using :math:`\\rm\\LaTeX`, which then, it takes longer to produce the
          plot.
        * If :math:`\\rm\\LaTeX` is not installed, it uses any available
          San-Serif font to render the plot.

       .. note::

           To manually disable interactive plot display, and save the plot as
           ``SVG`` instead, add the following in the very beginning of your
           code before importing ``imate``:

           .. code-block:: python

               >>> import os
               >>> os.environ['IMATE_NO_DISPLAY'] = 'True'

        :param correlation: A 2D array of coordinates of points. The
            correlation matrix is generated from the euclidean distance of the
            points. The ``points`` array has the shape
            ``(num_points, dimension)``.
        :type pointS: numpy-ndarray

        :param distance_scale: A parameter of correlation function that scales
            distance. It can be an array of the size of the dimension, which
            then it specifies a correlation for each dimension axis.
            Alternatively, it can be a scalar, which then it assumes an
            isotropic correlation scale for all dimension axes.
        :type distance_scale: float or numpy.ndarray

        :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
        :type nu: float

        :param sparse: Flag to indicate the correlation matrix should be
            sparse or dense matrix. If set to ``True``, you may also specify
            ``density``.
        :type parse: bool

        :param density: Specifies an approximate density of the non-zero
            elements of the generated sparse matrix. The actual density of the
            matrix may not be exactly the same as this value.
        :rtype: double

        :param plot: If ``True``, the matrix will be plotted.
        :type Plot: bool

        :param verbose: If ``True``, prints some information during the
            process.
        :type verbose: bool

        :return: Correlation matrix.
        :rtype: numpy.ndarray or scipy.sparse.csc

        **Example:**

        Generate a matrix of the shape ``(20,20)`` by mutual correlation of a
        set of :math:`20` points in the unit interval:

        .. code-block:: python

           >>> from imate import generate_matrix
           >>> A = generate_matrix(20)

        Generate a matrix of the shape :math:`(20^2, 20^2)` by mutual
        correlation of a grid of :math:`20 \\times 20` points in the unit
        square:

        .. code-block:: python

           >>> from imate import generate_matrix
           >>> A = generate_matrix(20, dimension=20)

        Generate a correlation matrix of shape ``(20, 20)`` based on 20 random
        points in unit square:

        .. code-block:: python

           >>> A = generate_matrix(size=20, dimension=20, grid=False)

        Generate a matrix of shape ``(20, 20)`` with spatial :math:`20` points
        that are more correlated:

        .. code-block:: python

           >>> A = generate_matrix(size=20, distance_scale=0.3)

        Sparsify correlation matrix of size :math:`(20^2, 20^2)` with
        approximate density of :math:`1e-3`

        .. code-block:: python

           >>> A = generate_matrix(size=20, dimension=2, sparse=True,
           ...                     density=1e-3)

        Plot a dense matrix of size :math:`(30^2, 30^2)` by

        .. code-block:: python

            >>> A = generate_matrix(size=30, dimension=2, plot=True)
        """

        # Compute the correlation between the set of points
        if sparse:

            # Generate a sparse matrix
            if derivative == 0:
                # This generates a new correlation matrix (no derivative).
                # The nnz of the matrix will be determined, and is not known
                # a priori.
                correlation_matrix = generate_sparse_correlation(
                    self.points,
                    self.distance_scale,
                    self.kernel,
                    derivative,
                    density,
                    verbose)

            else:
                # We use the same sparsity structure of self.K_der0 in the
                # derivative matrix.
                if self.K_der0 is None:
                    raise RuntimeError('To compute the derivative of a ' +
                                       'sparse correlation matrix, first, ' +
                                       'the correlation matrix itself ' +
                                       'should be computed.')

                # Generate derivative of correlation. The nnz of the matrix is
                # known a priori based on the zero-th derivative correlation
                # matrix that was calculated before. No new sparcity is
                # generated, rather, the sparsity structure of the matrix is
                # the same as self.K_der0.
                correlation_matrix = generate_sparse_correlation(
                    self.points,
                    self.distance_scale,
                    self.kernel,
                    derivative,
                    density,
                    verbose,
                    self.K_der0)

        else:

            # Generate a dense matrix
            correlation_matrix = generate_dense_correlation(
                self.points,
                self.distance_scale,
                self.kernel,
                derivative,
                verbose)

        # Plot Correlation Matrix
        if plot:
            self.plot_matrix(correlation_matrix, sparse, verbose)

        if derivative == 0:
            self.K_der0 = correlation_matrix
        elif derivative == 1:
            self.K_der1 = correlation_matrix
        elif derivative == 2:
            self.K_der2 = correlation_matrix
        else:
            raise ValueError('"derivative" should be 0, 1, or 2.')

    # ===========
    # plot matrix
    # ===========

    def plot_matrix(self, matrix, sparse, verbose=False):
        """
        Plots a given matrix.

        If the matrix is a sparse, it plots all non-zero elements with single
        color regardless of their values, and leaves the zero elements white.

        Whereas, if the matrix is not a sparse matrix, the colormap of the plot
        correspond to the value of the elements of the matrix.

        If a graphical backend is not provided, the plot is not displayed,
        rather saved as ``SVG`` file in the current directory of user.

        :param matrix: A 2D array
        :type matrix: numpy.ndarray or scipy.sparse.csc_matrix

        :param sparse: Determine whether the matrix is dense or sparse
        :type sparse: bool

        :param verbose: If ``True``, prints some information during the
            process.
        :type verbose: bool
        """

        # Load plot settings
        if plot_modules_exist:
            load_plot_settings()
        else:
            raise ImportError("Cannot load plot settings.")

        # Figure
        fig, ax = plt.subplots(figsize=(6, 4))

        if sparse:
            # Plot sparse matrix
            p = ax.spy(matrix, markersize=1, color='blue', rasterized=True)
        else:
            # Plot dense matrix
            p = ax.matshow(matrix, cmap='Blues')
            cbar = fig.colorbar(p, ax=ax)
            cbar.set_label('Correlation')

        ax.set_title('Correlation Matrix', y=1.11)
        ax.set_xlabel('Index $i$')
        ax.set_ylabel('Index $j$')

        plt.tight_layout()

        # Check if the graphical backend exists
        if matplotlib.get_backend() != 'agg':
            plt.show()
        else:
            # write the plot as SVG file in the current working directory
            save_plot(plt, 'CorrelationMatrix', transparent_background=True)