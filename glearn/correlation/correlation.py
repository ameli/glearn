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
import scipy
from ._dense_auto_correlation import dense_auto_correlation
from ._dense_cross_correlation import dense_cross_correlation
from ._sparse_auto_correlation import sparse_auto_correlation
from ._sparse_cross_correlation import sparse_cross_correlation
from ..kernels import Kernel, Matern
import imate
from ..priors.prior import Prior
from ..priors.uniform import Uniform
from .._utilities.timer import Timer

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
            scale=None,
            sparse=False,
            kernel_threshold=None,
            density=1e-3,
            verbose=False):
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

        # If points are 1d array, wrap them to a 2d array
        if points.ndim == 1:
            points = numpy.array([points], dtype=float).T

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
        self.kernel_threshold = kernel_threshold
        self.density = density
        self.verbose = verbose
        self.matrix_size = points.shape[0]
        self.dimension = points.shape[1]

        # Set distance scale. By initializing scale to None in the constructor,
        # it will be determined later as variable to the optimization problem.
        if numpy.isscalar(scale) or isinstance(scale, (list, numpy.ndarray)):
            # Input scale is given as a known numeric value.
            self.current_scale = None
            self.scale_prior = None

            # Setting current_scale
            self.set_scale(scale)

        elif isinstance(scale, Prior):
            # Input scale is in the form of a prior distribution.
            self.current_scale = None
            self.scale_prior = scale

        else:
            # Set scale to be a uniform improper prior if nothing is given.
            self.current_scale = None
            self.scale_prior = Uniform()

        # Determine whether a new matrix needs to be computed or not. Usually,
        # this is needed when (1) this class is initialized, and (2) when the
        # scale is changed. When current_scale_changed is True, the function
        # _update_matrix will generate a new correlation matrix.
        self.current_scale_changed = True

        # Initialize correlation matrix
        self.K_der0 = None
        self.K_der1 = None
        self.K_der2 = None

        # Initialize correlation affine matrix function (amf)
        self.K_amf_der0 = None
        self.K_amf_der1 = None
        self.K_amf_der2 = None

        # Initialize correlation matrix eigenvalues
        self.K_eig_der0 = None
        self.K_eig_der1 = None
        self.K_eig_der2 = None

        # Keeps which of the derivatives are updated (used only for sparse)
        self.K_der0_updated = False
        self.K_der1_updated = False
        self.K_der2_updated = False

        # Smallest and largest eigenvalues of K (only for zero-th derivative)
        self.K_eig_smallest = None
        self.K_eig_largest = None

        # Elapsed time of updating correlation (all updates combined). This
        # timer only keeps the time of correlation between training points and
        # themselves (no test point), during the training only.
        self.timer = Timer()

        # Keep number of how many times correlation matrix is computed/updated.
        # This counter is used only during the training only (no test point
        # correlation).
        self.num_cor_eval = 0

    # =========
    # get scale
    # =========

    def get_scale(self):
        """
        Returns the current scale of the correlation. The current scale is an
        actual numeric value, not a prior distribution function.
        """

        # if self.current_scale is None:
        #     raise ValueError('"scale" of correlation object is None.')

        return self.current_scale

    # =========
    # set scale
    # =========

    def set_scale(self, scale):
        """
        """

        # If the given scale is None, do not change the existing
        # self.current_scale attribute. This essentially leaves
        # self.current_scale unchanged. If the attribute self.current_scale is
        # also None, this should not happen.
        if scale is None:

            # If the attribute self.current_scale is also None, this should not
            # happen.
            if self.current_scale is None:
                raise ValueError('"scale" hyperparameter is undetermined.')

        else:

            # Here, scale is not None. Convert scale to array
            if numpy.isscalar(scale):
                if not isinstance(scale, (int, numpy.integer)) and \
                   not isinstance(scale, float):
                    raise TypeError('"scale" should be float.')

                # Convert distance scale to numpy array
                scale_ = numpy.array([scale], dtype=float)

            elif isinstance(scale, list):
                scale_ = numpy.array(scale)

            elif isinstance(scale, numpy.ndarray):
                scale_ = scale

            else:
                raise TypeError('"scale" should be either a scalar, a list ' +
                                'of numbers, or a numpy array.')

            # if scale is an array of length one, extend the array to
            # be the size of dimension
            dimension = self.points.shape[1]
            if scale_.size == 1:

                # Repeat correlation scale to an array of size dimension
                scale_ = numpy.repeat(scale_, dimension)

            elif scale_.size != dimension:
                # Check dimension matches the size of scale array
                raise ValueError('"scale" should have the same dimension as ' +
                                 'the "points".')

            # Check if self.scale should be updated
            if any(self.current_scale != scale_):
                self.current_scale = scale_
                self.current_scale_changed = True

    # ===============
    # get matrix size
    # ===============

    def get_matrix_size(self):
        """
        Returns the size of the matrix.
        """

        return self.matrix_size

    # ==========
    # get matrix
    # ==========

    def get_matrix(
            self,
            scale=None,
            derivative=[]):
        """
        Returns the correlation matrix. If the correlation is not available as
        a matrix, it generates the matrix from the kernel and spatial distance
        of the given points.
        """

        # Update matrix (if needed)
        self._update_matrix(scale, derivative)

        if len(derivative) == 0:
            return self.K_der0

        elif len(derivative) == 1:
            return self.K_der1[derivative[0]]

        elif len(derivative) == 2:
            return self.K_der2[derivative[0]][derivative[1]]

    # ==========================
    # get affine matrix function
    # ==========================

    def get_affine_matrix_function(
            self,
            scale=None,
            derivative=[]):
        """
        Returns an instance of ``imate.AffineMatrixFunction`` class.
        """

        # Update matrix (if needed)
        self._update_matrix(scale, derivative)

        if len(derivative) == 0:

            if self.K_amf_der0 is None or self.current_scale_changed:
                # Create new affine matrix function object
                self.K_amf_der0 = imate.AffineMatrixFunction(self.K_der0)

            return self.K_amf_der0

        elif len(derivative) == 1:

            if self.K_amf_der1 is None or self.current_scale_changed:
                # Create new affine matrix function object
                self.K_amf_der1 = [None] * self.dimension
                for p in range(self.dimension):
                    self.K_amf_der1[p] = imate.AffineMatrixFunction(
                            self.K_der1[derivative[0]])

            return self.K_amf_der1[derivative[0]]

        elif len(derivative) == 2:

            if self.K_amf_der2 is None or self.current_scale_changed:
                # Create new affine matrix function object
                self.K_amf_der2 = [[] for _ in range(self.dimension)]
                for p in range(self.dimension):
                    self.K_amf_der2[p] = [None] * self.dimension
                    for q in range(self.dimension):
                        self.K_amf_der2[p][q] = imate.AffineMatrixFunction(
                                self.K_der2[derivative[0]][derivative[1]])

            return self.K_amf_der2[derivative[0]][derivative[1]]

    # ===============
    # get eigenvalues
    # ===============

    def get_eigenvalues(
            self,
            scale=None,
            derivative=[]):
        """
        Returns the eigenvalues of the correlation matrix or its derivative.
        """

        if self.sparse:
            raise RuntimeError('When the correlation matrix is sparse, ' +
                               'the "imate_method" cannot be set to ' +
                               '"eigenvalue". You may set ' +
                               '"imate_method" to "cholesky", "slq", or ' +
                               '"hutchinson."')

        # Update matrix (if needed)
        self._update_matrix(scale, derivative)

        if len(derivative) == 0:

            if self.K_eig_der0 is None or self.current_scale_changed:
                self.K_eig_der0 = scipy.linalg.eigh(
                        self.K_der0, eigvals_only=True, check_finite=False)

            return self.K_eig_der0

        elif len(derivative) == 1:

            if self.K_eig_der1 is None or self.current_scale_changed:
                self.K_eig_der1 = [None] * self.dimension
                for p in range(self.dimension):
                    self.K_eig_der1[p] = scipy.linalg.eigh(
                        self.K_der1[p], eigvals_only=True, check_finite=False)

            return self.K_eig_der1[derivative[0]]

        elif len(derivative) == 2:

            if self.K_eig_der2 is None or self.current_scale_changed:
                self.K_eig_der2 = [[] for _ in range(self.dimension)]
                for p in range(self.dimension):
                    self.K_eig_der2[p] = [None] * self.dimension
                    for q in range(self.dimension):
                        self.K_eig_der2[p][q] = scipy.linalg.eigh(
                                self.K_der2[p][q], eigvals_only=True,
                                check_finite=False)

            return self.K_eig_der2[derivative[0]][derivative[1]]

    # =======================
    # get extreme eigenvalues
    # =======================

    def get_extreme_eigenvalues(
            self,
            scale=None):
        """
        Returns the smallest and the largest eigenvalues of K (only for zero-th
        derivative).
        """

        # Update matrix (if needed)
        self._update_matrix(scale)

        if self.K_eig_smallest is None or self.K_eig_largest is None or \
                self.K_der0 is None or self.current_scale_changed:

            n = self.matrix_size

            # Compute smallest eigenvalue
            if self.sparse:
                self.K_eig_smallest = scipy.sparse.linalg.eigsh(
                        self.K_der0, k=1, which='SM', return_eigenvector=False)
            else:
                self.K_eig_smallest = scipy.linalg.eigh(
                        self.K_der0, eigvals_only=True, check_finite=False,
                        subset_by_index=[0, 0])[0]

            # Compute largest eigenvalue
            if self.sparse:
                self.K_eig_smallest = scipy.sparse.linalg.eigsh(
                        self.K_der0, k=1, which='LM', return_eigenvector=False)
            else:
                self.K_eig_largest = scipy.linalg.eigh(
                        self.K_der0, eigvals_only=True, check_finite=False,
                        subset_by_index=[n-1, n-1])[0]

        return self.K_eig_smallest, self.K_eig_largest

    # =============
    # update matrix
    # =============

    def _update_matrix(
            self,
            scale=None,
            derivative=[]):
        """
        If the matrix has not been generated before, or if the matrix settings
        has changed, this function generates a new matrix. It returns the
        status of whether a new matrix generated or not.
        """

        # Check arguments
        if len(derivative) not in (0, 1, 2):
            raise ValueError('"derivative" order should be 0, 1, or 2.')

        # If the given scale is different than self.current_scale, the function
        # below will update self.current_scale. Also, it will set
        # self.current_scale_changed to True.
        self.set_scale(scale)

        # Determine whether the matrix or its derivative should be generated
        update_needed = False
        if self.current_scale_changed:
            update_needed = True
        elif (len(derivative) == 0) and (self.K_der0 is None):
            update_needed = True
        elif (len(derivative) == 1) and (self.K_der1 is None):
            update_needed = True
        elif (len(derivative) == 2) and (self.K_der2 is None):
            update_needed = True
        elif (len(derivative) == 0) and (not self.K_der0_updated):
            update_needed = True
        elif (len(derivative) == 1) and (not self.K_der1_updated):
            update_needed = True
        elif (len(derivative) == 2) and (not self.K_der2_updated):
            update_needed = True

        # Generate new correlation matrix
        if update_needed:

            # Keep time and count of updates
            self.num_cor_eval += 1
            self.timer.tic()

            # Sparse matrix of derivative 1 and 2 needs matrix of derivative 0
            if (len(derivative) > 0) and self.sparse and (self.K_der0 is None):
                # Before generating matrix of derivative 1 or 2, first,
                # generate correlation matrix of derivative 0
                no_derivative = []
                self._generate_correlation_matrix(
                        self.current_scale, no_derivative)

            # The main line where new matrix is generated
            self._generate_correlation_matrix(self.current_scale, derivative)

            # End if extensive computation
            self.timer.toc()

            # if scale was changed, all matrices should be recomputed
            if self.current_scale_changed:
                self.K_der0_updated = False
                self.K_der1_updated = False
                self.K_der2_updated = False

            # Specify which derivative was updated
            if len(derivative) == 0:
                self.K_der0_updated = True
            elif len(derivative) == 1:
                self.K_der1_updated = True
            elif len(derivative) == 2:
                self.K_der2_updated = True

            # If scale was changed, all eigenvalues and amf have to be
            # recomputed again. So, we set them to None to signal other
            # functions that they need to be recomputed.
            if self.current_scale_changed:

                # Affine matrix functions
                self.K_amf_der0 = None
                self.K_amf_der1 = None
                self.K_amf_der2 = None

                # Eigenvalues
                self.K_eig_der0 = None
                self.K_eig_der1 = None
                self.K_eig_der2 = None

            # Indicate that update has been done
            self.current_scale_changed = False

    # ===========================
    # generate correlation matrix
    # ===========================

    def _generate_correlation_matrix(
            self,
            scale,
            derivative):
        """
        Generates auto-correlation matrix between training points and
        themselves. This matrix is square.
        """

        if len(derivative) > 2:
            raise ValueError('"derivative" order should be 0, 1, or 2.')

        # Compute the correlation between the set of points
        if self.sparse:

            # Generate a sparse matrix
            if len(derivative) == 0:
                # This generates a new correlation matrix (no derivative).
                # The nnz of the matrix will be determined, and is not known
                # a priori.
                correlation_matrix = sparse_auto_correlation(
                    self.points, scale, self.kernel, derivative,
                    self.kernel_threshold, self.density, test_points=None,
                    correlation_matrix=None, verbose=self.verbose)

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
                correlation_matrix = sparse_auto_correlation(
                    self.points, scale, self.kernel, derivative,
                    self.kernel_threshold, self.density, test_points=None,
                    correlation_matrix=self.K_der0, verbose=self.verbose)

        else:

            # Generate a dense matrix
            correlation_matrix = dense_auto_correlation(
                self.points, scale, self.kernel, derivative)

        if len(derivative) == 0:
            self.K_der0 = correlation_matrix
        elif len(derivative) == 1:
            self.K_der1 = correlation_matrix
        elif len(derivative) == 2:
            self.K_der2 = correlation_matrix

    # ================
    # auto correlation
    # ================

    def auto_correlation(self, test_points):
        """
        Computes the auto-correlation between the test points and themselves.
        The output is a square, symmetric, and positive-semi definite matrix.
        Because the correlation is computed between a set of points and
        themselves, this generating correlation with this function is twice
        faster than using cross_correlation.
        """

        derivative = []

        # Compute the correlation between the set of points
        if self.sparse:

            # This generates a new correlation matrix (no derivative). The nnz
            # of the matrix will be determined, and is not known a priori.
            correlation_matrix = sparse_auto_correlation(
                test_points, self.current_scale, self.kernel, derivative,
                self.kernel_threshold, self.density, test_points=None,
                correlation_matrix=None, verbose=self.verbose)

        else:

            # Generate a dense matrix
            correlation_matrix = dense_auto_correlation(
                test_points, self.current_scale, self.kernel, derivative,
                test_points=None)

        return correlation_matrix

    # =================
    # cross correlation
    # =================

    def cross_correlation(self, test_points):
        """
        Computes the cross-correlation between the training points (points
        which this object is initialized with), and a given set of test points.
        This matrix is rectangular.
        """

        # Compute the correlation between the set of points
        if self.sparse:

            # This generates a new correlation matrix (no derivative).
            # The nnz of the matrix will be determined, and is not known
            # a priori.
            correlation_matrix = sparse_cross_correlation(
                self.points, test_points, self.current_scale, self.kernel,
                self.kernel_threshold, self.density, verbose=self.verbose)

        else:

            # Generate a dense matrix
            correlation_matrix = dense_cross_correlation(
                self.points, test_points, self.current_scale, self.kernel)

        return correlation_matrix

    # ====
    # plot
    # ====

    def plot(self, derivative=[]):
        """
        Plots the (auto) correlation matrix, which it the correlation matrix
        between self.points and themselves.

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

        if self.current_scale is None:
            raise ValueError('Correlation cannot be plotted if "scale" is ' +
                             'not given.')

        # Load plot settings
        if plot_modules_exist:
            load_plot_settings()
        else:
            raise ImportError("Cannot load plot settings.")

        # Get correlation matrix
        matrix = self.get_matrix(derivative=derivative)

        # Figure
        fig, ax = plt.subplots(figsize=(6, 4))

        if self.sparse:
            # Plot sparse matrix
            p = ax.spy(matrix, markersize=1, color='blue', rasterized=True)
        else:
            # Plot dense matrix
            p = ax.matshow(matrix, cmap='Blues', vmin=0.0, vmax=1.0)
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
            save_plot(plt, 'correlation', transparent_background=True)
