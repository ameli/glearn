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
import scipy
from scipy.sparse import isspmatrix
from .._correlation import Correlation
from ._mixed_correlation import MixedCorrelation
import imate

__all__ = ['Covariance']


# ==========
# Covariance
# ==========

class Covariance(object):
    """
    Create mixed covariance model.

    This class creates mixed-covariance model which can be defined by a set of
    known or unknown hyperparameters and kernel functions. The covariance
    object can compute:

    * The auto-covariance or cross-covariance between a set of training
      and test points, or the derivative of the covariance with respect to the
      set of hyperparameters.
    * The covariance object can also compute basic matrix functions of the
      covariance matrix, such as log-determinant, or the trace of the functions
      of the matrix.
    * Solve a linear system or perform matrix-matrix or matrix-vector
      multiplication involving the covariance matrix or its derivatives with
      respect to hyperparameters.

    Parameters
    ----------

    x : numpy.ndarray
        A 2D array of data points where each row of the array is the coordinate
        of a point :math:`\\boldsymbol{x} = (x_1, \\dots, x_d)`. The
        array size is :math:`n \\times d` where :math:`n` is the number of the
        points and :math:`d` is the dimension of the space of points.

    sigma : float, default=None
        The hyperparameter :math:`\\sigma` of the covariance model where
        :math:`\\sigma^2` represents the variance of the correlated errors of
        the model. :math:`\\sigma` should be positive. If `None` is given, an
        optimal value for :math:`\\sigma` is found during the training process.

    sigma0 : float, default=None
        The hyperparameter :math:`\\varsigma` of the covariance model where
        :math:`\\varsigma^2` represents the variance of the input noise to the
        model. :math:`\\varsigma` should be positive. If `None` is given, an
        optimal value for :math:`\\varsigma` is found during the training
        process.

    scale : float or array_like[float], default=None
        The scale hyperparameters
        :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_d)` in
        scales the distance between data points in :math:`\\mathbb{R}^d`. If an
        array of the size :math:`d` is given, each :math:`\\alpha_i` scales the
        distance in the :math:`i`-th dimension. If a scalar value
        :math:`\\alpha` is given, all dimensions are scaled isometrically.
        If set to `None`, optimal values of the scale hyperparameters are found
        during the training process by the automatic relevance determination
        (ARD).

    kernel : glearn.kernels.Kernel, default=glearn.kernels.Matern
        The correlation kernel :math:`k` that generates the correlation matrix
        :math:`\\mathbf{K}`. This argument should be an instance of one of the
        derived classes of :class:`glearn.kernels.Kernel`. If `None`, the
        Matern kernel :class:`glearn.kernels.Matern` is used.

    kernel_threshold : float, default=None,
        The threshold :math:`\\tau` to taper the kernel function. Namely,
        the kernel values :math:`k < \\tau` are set to zero. This is used to
        decorrelate data points that are away from each other by a distance,
        yielding a *sparse* correlation matrix of the data points. This option
        is relevant if ``sparse`` is set to `True`.

    sparse : bool, default=False
        It `True`, it sparsifies the correlation matrix :math:`\\mathbf{K}` and
        hence, the covariance matrix :math:`\\boldsymbol{\\Sigma}` using
        kernel tapering (see ``kernel_threshold`` and ``density``).

    density : float, default=1e-3,
        Sets an approximate density of sparse matrices. This argument is
        another way (along with ``kernel_threshold``) to specify the sparsity
        of the covariance matrix. The matrix density is the  This option is
        relevant if ``sparse`` is set to `True`.

        .. note::

            This option only sets an *approximate* density of the covariance
            matrix. The actual matrix density may be slightly different than
            the specified value.

    imate_options : dict, default={'method': 'cholesky'}
        The internal computations of the functions
        :meth:`glearn.Covariance.logdet`, :meth:`glearn.Covariance.trace`, and
        :meth:`glearn.Covariance.traceinv` are performed by
        `imate <https://ameli.github.io/imate/index.html>`_ package. This
        argument can pass a dictionary of options to pass to the corresponding
        functions of the `imate` package. See
        `API Reference <https://ameli.github.io/imate>`_ of `imate` package
        for details.

    interpolate : bool, default=False
        If `True`, the matrix functions
        :meth:`glearn.Covariance.logdet`, :meth:`glearn.Covariance.trace`, and
        :meth:`glearn.Covariance.traceinv` for the mixed covariance function
        are interpolated with respect to the
        hyperparameters :math:`\\sigma` and :math:`\\varsigma`. See [1]_ for
        details. This approach can yield a significant speed up during the
        training process but with the loss of accuracy.

    tol : float, default=1e-8
        The tolerance of error of solving the linear systems using conjugate
        gradient method used in :meth:`glearn.Covariance.solve` function.

    verbose : bool, default=False
        It `True`, verbose output is printed during the computation.

    Attributes
    ----------

    cor : glearn._correlation.Correlation
        An object representing the correlation matrix :math:`\\mathbf{K}`.

    cor : glearn._covariance.MixedCorrelation
        An object representing the mixed correlation matrix
        :math:`\\mathbf{K} + \\eta \\mathbf{I}`.

    Methods
    -------

    get_size
    get_imate_options
    set_imate_options
    set_scale
    get_scale
    set_sigmas
    get_sigmas
    get_matrix
    trace
    traceinv
    logdet
    solve
    dot
    auto_covariance
    cross_covariance

    See Also
    --------

    glearn.GaussianProcess

    Notes
    -----

    **Regression Model:**

    A regression model to fit the data :math:`y = f(\\boldsymbol{x})`
    for the points :math:`\\boldsymbol{x} \\in \\mathcal{D} \\in \\mathbb{R}^d`
    and data :math:`y \\in \\mathbb{R}` is

    .. math::

        f(\\boldsymbol{x}) = \\mu(\\boldsymbol{x}) + \\delta(\\boldsymbol{x})
        + \\epsilon,

    where

    * :math:`\\mu` is a deterministic mean function.
    * :math:`\\delta` is a zero-mean stochastic function representing the
      missfit of the regression model.
    * :math:`\\epsilon` is a zero-mean stochastic function representing the
      input noise.

    **Covariance of Regression:**

    The covariance of the stochastic function :math:`\\delta` on discrete
    data points :math:`\\{ \\boldsymbol{x}_i \\}_{i=1}^n` is the
    :math:`n \\times n` covariance matrix

    .. math::

        \\sigma^2 \\mathbf{K} =
        \\mathbb{E}[\\delta(\\boldsymbol{x}_i), \\delta(\\boldsymbol{x}_j)],

    where :math:`\\sigma^2` is the variance and :math:`\\mathbf{K}` is
    considered as the correlation matrix.

    Similarly, the covariance of the stochastic function :math:`\\epsilon`is
    the :math:`n \\times n` covariance matrix

    .. math::

        \\varsigma^2 \\mathbf{I} = \\mathbb{E}[\\epsilon, \\epsilon],

    where :math:`\\varsigma^2` is the variance of noise and
    :math:`\\mathbf{I}` is the identity matrix.

    The overall mixed-covariance model for the linear model :math:`f` is

    .. math::

        \\boldsymbol{\\Sigma}(\\sigma^2, \\varsigma^2, \\boldsymbol{\\alpha}) =
        \\sigma^2 \\mathbf{K} + \\varsigma^2 \\mathbf{I}.

    References
    ----------

    .. [1] Ameli, S., and Shadden. S. C. (2022). *Interpolating Log-Determinant
           and Trace of the Powers of Matrix*
           :math:`\\mathbf{A} + t\\mathbf{B}`. `arXiv: 2009.07385
           <https://arxiv.org/abs/2207.08038>`_ [math.NA].

    Examples
    --------

    **Create Covariance Object:**

    Create a covariance matrix based on a set of sample data with four
    points in :math:`d=2` dimensional space.

    .. code-block:: python
        :emphasize-lines: 7

        >>> # Generate a set of points
        >>> from glearn.sample_data import generate_points
        >>> x = generate_points(num_points=4, dimension=2)

        >>> # Create a covariance object
        >>> from glearn import Covariance
        >>> cov = Covariance(x)

    By providing a set of hyperparameters, the covariance matrix can be
    fully defined. Here we set :math:`\\sigma=2`, :math:`\\varsigma=3`, and
    :math:`\\boldsymbol{\\alpha}= (1, 2)`.

    .. code-block:: python

        >>> # Get the covariance matrix for given hyperparameters
        >>> cov.set_sigmas(2.0, 3.0)
        >>> cov.set_scale([1.0, 2.0])
        >>> cov.get_matrix()
        array([[13.        ,  3.61643745,  3.51285267,  3.47045163],
               [ 3.61643745, 13.        ,  3.32078482,  3.14804532],
               [ 3.51285267,  3.32078482, 13.        ,  3.53448631],
               [ 3.47045163,  3.14804532,  3.53448631, 13.        ]])

    **Specify Hyperparameter at Instantiation:**

    The hyperparameters can also be defined at the time of instantiating the
    covariance object.

    .. code-block:: python

        >>> # Create a covariance object
        >>> cov.traceinv(sigma=2.0, sigma0=3.0, scale=[1.0, 2.0])

    **Specify Correlation Kernel:**

    The kernel function that creates the correlation matrix :math:`\\mathbf{K}`
    can be specified by one of the kernel objects derived from
    :class:`glearn.kernels.Kernel` class. For instance, in the next example, we
    set a square exponential kernel :class:`glearn.kernels.SquareExponential`.

    .. code-block:: python

        >>> # Create a kernel object
        >>> from glearn import kernels
        >>> kernel = kernels.SquareExponential()

        >>> # Create covariance object with the above kernel
        >>> cov.traceinv(kernel=kernel)

    **Sparse Covariance:**

    The covariance object can be configured to
    """

    # ====
    # init
    # ====

    def __init__(
            self,
            x,
            sigma=None,
            sigma0=None,
            scale=None,
            kernel=None,
            kernel_threshold=None,
            sparse=False,
            density=1e-3,
            imate_options={'method': 'cholesky'},
            interpolate=False,
            tol=1e-8,
            verbose=False):
        """
        """

        # The rest of argument will be checked in self.cor
        self._check_arguments(sigma, sigma0, tol)

        # Set attributes
        self.sigma = sigma
        self.sigma0 = sigma0
        self.tol = tol

        # Correlation (matrix K)
        self.cor = Correlation(x, kernel=kernel, scale=scale, sparse=sparse,
                               kernel_threshold=kernel_threshold,
                               density=density, verbose=verbose)

        # Mixed correlation (K + eta I)
        self.mixed_cor = MixedCorrelation(self.cor, interpolate=interpolate,
                                          imate_options=imate_options)

    # ===============
    # Check arguments
    # ===============

    def _check_arguments(self, sigma, sigma0, tol):
        """
        """

        # Check tol
        if not isinstance(tol, float):
            raise TypeError('"tol" should be a float number.')
        elif tol < 0.0:
            raise ValueError('"tol" should be non-negative.')

        # Check sigma
        if sigma is not None:
            if not isinstance(sigma, int) and not isinstance(sigma, float):
                raise TypeError('"sigma" should be a float type.')
            elif sigma < 0.0:
                raise ValueError('"sigma" cannot be negative.')

        # Check sigma0
        if sigma0 is not None:
            if not isinstance(sigma0, int) and not isinstance(sigma0, float):
                raise TypeError('"sigma0" should be a float type.')
            elif sigma0 < 0.0:
                raise ValueError('"sigma0" cannot be negative.')

    # ========
    # get size
    # ========

    def get_size(self):
        """
        Returns the size of the covariance matrix.

        Returns
        -------

        n : int
            The size :math:`n` of the :math:`n \\times n` covariance matrix.

        See Also
        --------

        glearn.Covariance.get_matrix

        Examples
        --------

        Create a covariance matrix based on a set of sample data with four
        points in :math:`d=2` dimensional space.

        .. code-block:: python
            :emphasize-lines: 10

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=4, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x)

            >>> # Get the size of the covariance matrix
            >>> cov.get_size()
            4

        The size of the covariance defined in the above is the same as the size
        of the training points, ``x.shape[0]``.

        By providing a set of hyperparameters, the covariance matrix can be
        fully defined. Here we set :math:`\\sigma=2`, :math:`\\varsigma=3`, and
        :math:`\\boldsymbol{\\alpha}= (1, 2)`.

        .. code-block:: python

            >>> # Get the covariance matrix for given hyperparameters
            >>> cov.set_sigmas(2.0, 3.0)
            >>> cov.set_scale([1.0, 2.0])
            >>> cov.get_matrix()
            array([[13.        ,  3.61643745,  3.51285267,  3.47045163],
                   [ 3.61643745, 13.        ,  3.32078482,  3.14804532],
                   [ 3.51285267,  3.32078482, 13.        ,  3.53448631],
                   [ 3.47045163,  3.14804532,  3.53448631, 13.        ]])
        """

        return self.mixed_cor.cor.points.shape[0]

    # =================
    # get imate options
    # =================

    def get_imate_options(self):
        """
        Returns the dictionary of options that is passed to the imate package.

        Returns
        -------

        imate_options : dict
            A dictionary of options to be passed to the functions in
            `imate <https://ameli.github.io/imate/index.html>`_ package.

        Examples
        --------

        Create a covariance object and set ``imate_options``:

        .. code-block:: python
            :emphasize-lines: 15

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=30, dimension=1)

            >>> # Define imate options
            >>> options = {
            ...     'method': 'cholesky',
            ... }

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x, imate_options=options)

            >>> # Get default imate options
            >>> cov.get_imate_options()
            {
                'method': 'cholesky'
            }

        Now, change the ``imate_options`` of the above covariance object:

        .. code-block:: python
            :emphasize-lines: 10

            >>> # Define new imate options
            >>> options = {
            ...     'method': 'slq',
            ...     'min_num_samples': 30,
            ...     'max_num_samples': 100
            ... }
            >>> cov.set_imate_options(options)

            >>> # Check again the updated options
            >>> cov.get_imate_options()
            {
                'method': 'slq',
                'min_num_samples': 30,
                'max_num_samples': 100
            }
        """

        return self.mixed_cor.imate_options

    # =================
    # set imate options
    # =================

    def set_imate_options(self, imate_options):
        """
        Updates the dictionary of options that is passed to the imate package.

        .. note::

            This function is intended to be used internally.

        Parameters
        ----------

        imate_options : dict
            A dictionary of options to be passed to the functions in
            `imate <https://ameli.github.io/imate/index.html>`_ package.

        See Also
        --------

        glearn.Covariance.set_imate_options

        Notes
        -----

        This function updates the attribute ``imate_options`` which is a
        dictionary of options configuring the functions of the imate package.
        The existing options in the dictionary ``imate_option`` are
        overwritten, and new options will be added (if they do not already
        exist in the current dictionary).

        Examples
        --------

        Create a covariance object and set ``imate_options``:

        .. code-block:: python

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=30, dimension=1)

            >>> # Define imate options
            >>> options = {
            ...     'method': 'cholesky',
            ... }

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x, imate_options=options)

            >>> # Check default imate options
            >>> cov.get_imate_options()
            {
                'method': 'cholesky'
            }

        Now, change the ``imate_options`` of the above covariance object:

        .. code-block:: python
            :emphasize-lines: 7

            >>> # Define new imate options
            >>> options = {
            ...     'method': 'slq',
            ...     'min_num_samples': 30,
            ...     'max_num_samples': 100
            ... }
            >>> cov.set_imate_options(options)

            >>> # Check again the updated options
            >>> cov.get_imate_options()
            {
                'method': 'slq',
                'min_num_samples': 30,
                'max_num_samples': 100
            }
        """

        # If method key does not exists, set a default with Cholesky method.
        if 'method' not in imate_options:
            imate_options['method'] = 'cholesky'

        self.mixed_cor.imate_options = imate_options

    # =========
    # set scale
    # =========

    def set_scale(self, scale):
        """
        Sets the array of scale hyperparameters of the correlation matrix.

        Parameters
        ----------

        scale : float or array_like[float], default=None
            The scale hyperparameters
            :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_d)` in
            scales the distance between data points in :math:`\\mathbb{R}^d`.
            If an array of the size :math:`d` is given, each :math:`\\alpha_i`
            scales the distance in the :math:`i`-th dimension. If a scalar
            value :math:`\\alpha` is given, all dimensions are scaled
            isometrically. If set to `None`, optimal values of the scale
            hyperparameters are found during the training process by the
            automatic relevance determination (ARD).

        See Also
        --------

        glearn.Covariance.get_imate_options

        Examples
        --------

        Set scales :math:`\\boldsymbol{\\alpha} = [2, 3]` for a two-dimensional
        data:

        .. code-block:: python
            :emphasize-lines: 12

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=20, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x)
            >>> cov.get_scale()
            None

            >>> # Change scale
            >>> cov.set_scale([2.0, 3.0])
            >>> cov.get_scale()
            array([2., 3.])
        """

        self.mixed_cor.set_scale(scale)

    # =========
    # get scale
    # =========

    def get_scale(self):
        """
        Returns the array of scale hyperparameters of the correlation matrix.

        Returns
        -------

        scale : float or array_like[float]
            The scale hyperparameters
            :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_d)`.

        See Also
        --------

        glearn.Covariance.set_scale
        glearn.Covariance.get_sigmas

        Examples
        --------

        Set scales :math:`\\boldsymbol{\\alpha} = [2, 3]` for a two-dimensional
        data:

        .. code-block:: python
            :emphasize-lines: 8, 13

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=20, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x)
            >>> cov.get_scale()
            None

            >>> # Change scale
            >>> cov.set_scale([2.0, 3.0])
            >>> cov.get_scale()
            array([2., 3.])
        """

        return self.mixed_cor.get_scale()

    # ==========
    # set sigmas
    # ==========

    def set_sigmas(self, sigma, sigma0):
        """
        Sets :math:`\\sigma` and :math:`\\varsigma` hyperparameters of the
        covariance model.

        Parameters
        ----------

        sigma : float, default=None
            The hyperparameter :math:`\\sigma` of the covariance model where
            :math:`\\sigma^2` represents the variance of the correlated errors
            of the model. :math:`\\sigma` should be positive. If `None` is
            given, an optimal value for :math:`\\sigma` is found during the
            training process.

        sigma0 : float, default=None
            The hyperparameter :math:`\\varsigma` of the covariance model where
            :math:`\\varsigma^2` represents the variance of the input noise to
            the model. :math:`\\varsigma` should be positive. If `None` is
            given, an optimal value for :math:`\\varsigma` is found during the
            training process.

        See Also
        --------

        glearn.Covariance.get_sigmas
        glearn.Covariance.set_scale

        Notes
        -----

        .. note::

            After training process when optimal values of the hyperparameters
            :math:`\\sigma` and :math:`\\varsigma` is obtained, this function
            is automatically called to update these hyperparameters as the
            attributes of the covariance class.

        Examples
        --------

        Set hyperparameters :math:`\\sigma = 2` and :math:`\\varsigma = 3`.

        .. code-block:: python
            :emphasize-lines: 12

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=20, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x, sigma=0.0, sigma0=1.0)
            >>> cov.get_sigmas()
            (0.0, 1.0)

            >>> # Change sigmas
            >>> cov.set_sigmas(2.0, 3.0)
            >>> cov.get_sigmas()
            (2.0, 3.0)
        """

        if sigma is None:
            raise ValueError('"sigma" cannot be None.')
        if sigma0 is None:
            raise ValueError('"sigma0" cannot be None.')

        self.sigma = sigma
        self.sigma0 = sigma0

        # Set eta for mixed_cor object
        self.mixed_cor.set_eta(self.sigma, self.sigma0)

    # ==========
    # get sigmas
    # ==========

    def get_sigmas(self, sigma=None, sigma0=None):
        """
        Returns :math:`\\sigma` and :math:`\\varsigma` hyperparameters of the
        covariance model.

        Parameters
        ----------

        sigma : float, default=None
            The hyperparameter :math:`\\sigma` of the covariance model where
            :math:`\\sigma^2` represents the variance of the correlated errors
            of the model. :math:`\\sigma` should be positive. If `None` is
            given, an optimal value for :math:`\\sigma` is found during the
            training process.

        sigma0 : float, default=None
            The hyperparameter :math:`\\varsigma` of the covariance model where
            :math:`\\varsigma^2` represents the variance of the input noise to
            the model. :math:`\\varsigma` should be positive. If `None` is
            given, an optimal value for :math:`\\varsigma` is found during the
            training process.

        See Also
        --------

        glearn.Covariance.get_sigmas
        glearn.Covariance.set_scale

        Notes
        -----

        After training process when optimal values of the hyperparameters
        :math:`\\sigma` and :math:`\\varsigma` is obtained, this function can
        be used to return these hyperparameters.

        Examples
        --------

        Set hyperparameters :math:`\\sigma = 2` and :math:`\\varsigma = 3`.

        .. code-block:: python
            :emphasize-lines: 8, 13

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=20, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x, sigma=0.0, sigma0=1.0)
            >>> cov.get_sigmas()
            (0.0, 1.0)

            >>> # Change sigmas
            >>> cov.set_sigmas(2.0, 3.0)
            >>> cov.get_sigmas()
            (2.0, 3.0)
        """

        # Get sigma
        if sigma is None:
            if self.sigma is None:
                raise ValueError('"sigma" cannot be None.')
            else:
                sigma = self.sigma

        # Get sigma0
        if sigma0 is None:
            if self.sigma0 is None:
                raise ValueError('"sigma0" cannot be None.')
            else:
                sigma0 = self.sigma0

        return sigma, sigma0

    # ==========
    # get matrix
    # ==========

    def get_matrix(
            self,
            sigma=None,
            sigma0=None,
            scale=None,
            derivative=[]):
        """
        Compute the covariance matrix or its derivatives for a given set of
        hyperparameters.

        Parameters
        ----------

        sigma : float, default=None
            The hyperparameter :math:`\\sigma` of the covariance model where
            :math:`\\sigma^2` represents the variance of the correlated errors
            of the model. :math:`\\sigma` should be positive and cannot be
            `None`.

        sigma0 : float, default=None
            The hyperparameter :math:`\\varsigma` of the covariance model where
            :math:`\\varsigma^2` represents the variance of the input noise to
            of the model. :math:`\\sigma` should be positive and cannot be
            `None`.

        scale : float or array_like[float], default=None
            The scale hyperparameters
            :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_d)` in
            scales the distance between data points in :math:`\\mathbb{R}^d`.
            If an array of the size :math:`d` is given, each :math:`\\alpha_i`
            scales the distance in the :math:`i`-th dimension. If a scalar
            value :math:`\\alpha` is given, all dimensions are scaled
            isometrically. :math:`\\boldsymbol{\\alpha}` cannot be `None`.

        derivative : list, default=[]
            Specifies a list of derivatives of covariance matrix with respect
            to the hyperparameters :math:`\\boldsymbol{\\alpha} = (\\alpha_1,
            \\dots, \\alpha_d)`. A list of the size :math:`q` with the
            components ``[i, j, ..., k]`` corresponds to take the derivative

            .. math::

                \\left. \\frac{\\partial^q}{\\partial \\alpha_{i+1} \\partial
                \\alpha_{j+1} \\dots \\partial \\alpha_{k+1}}
                \\boldsymbol{\\Sigma}(\\boldsymbol{\\alpha} \\vert
                \\sigma^2, \\varsigma^2) \\right|_{\\boldsymbol{\\alpha}}.

            .. note::

                The derivative with respect to each hyperparameter
                :math:`\\alpha_i` can be at most of the order two,
                :math:`\\partial^2 / \\partial \\alpha_i^2`. That is, each
                index in the ``derivative`` list can appear at most twice.
                For instance ``derivative=[1, 1]`` (second order derivative
                with respect to :math:`\\alpha_{2}`) is a valid input argument,
                how ever ``derivative=[1, 1, 1]`` (third order derivative) is
                an invalid input.

        Returns
        -------

        S : numpy.ndarray
            An array of the size :math:`n \\times \\times n` where :math:`n` is
            the size of the matrix.

        See Also
        --------

        glearn.Covariance.get_size
        glearn.Covariance.auto_covariance
        glearn.Covariance.cross_covariance

        Notes
        -----

        This function returns

        .. math::

            \\left. \\frac{\\partial^q}{\\partial \\alpha_{i} \\partial
            \\alpha_{j} \\dots \\partial \\alpha_{k}}
            \\boldsymbol{\\Sigma}(\\boldsymbol{\\alpha} \\vert
            \\sigma, \\varsigma) \\right|_{\\boldsymbol{\\alpha}},

        where the covariance matrix :math:`\\boldsymbol{\\Sigma}` is defined by

        .. math::

            \\boldsymbol{\\Sigma}(\\boldsymbol{\\alpha}, \\sigma, \\varsigma) =
            \\sigma^2 \\mathbf{K}(\\boldsymbol{\\alpha}) + \\varsigma^2
            \\mathbf{I}.

        In the above, :math:`\\mathbf{I}` is the identity matrix and
        :math:`\\mathbf{K}` is the correlation matrix that depends on a set of
        scale hyperparameters :math:`\\boldsymbol{\\alpha}=(\\alpha_1, \\dots,
        \\alpha_d)`.

        **Derivatives:**

        Note that the indices in list ``derivative=[i, j, ..., k]`` are
        zero-indexed, meaning that the index ``i`` corresponds to take
        derivative with respect to the hyperparameter :math:`\\alpha_{i+1}`.
        For instance:

        * ``[]`` corresponds to no derivative.
        * ``[0]`` corresponds to :math:`\\partial / \\partial \\alpha_1` and
          ``[1]`` corresponds to :math:`\\partial / \\partial
          \\alpha_2`.
        * ``[0, 2]`` corresponds to :math:`\\partial^2 /
          \\partial \\alpha_1 \\partial \\alpha_3`.
        * ``[0, 0]`` corresponds to :math:`\\partial^2 /
          \\partial \\alpha_1^2`.
        * ``[0, 2, 2, 4]`` corresponds to :math:`\\partial^4 /
          \\partial \\alpha_1 \\partial \\alpha_{3}^2 \\partial \\alpha_5`.

        **Output Matrix:**

        The covariance matrix :math:`\\boldsymbol{\\Sigma}` or its derivatives
        is symmetric, i.e., :math:`\\partial \\Sigma_{ij} = \\partial
        \\Sigma_{ji}`. Also, the covariance matrix is positive-definite,
        however, its derivatives are not necessarily positive-definite. In
        addition, the diagonal elements of the derivatives of the covariance
        matrix are zero, i.e., :math:`\\partial \\Sigma_{ii} = 0`.

        Examples
        --------

        **Basic Usage:**

        Create a sample dataset with four points in :math:`d=2` dimensional
        space. Then, compute the covariance matrix
        :math:`\\boldsymbol{\\Sigma} (\\boldsymbol{\\alpha}, \\sigma,
        \\varsigma)` for :math:`\\boldsymbol{\\alpha} = (1, 2)`,
        :math:`\\sigma=2`, and :math:`\\varsigma=3`.

        .. code-block:: python
            :emphasize-lines: 10

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=4, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x)

            >>> # Compute covariance matrix
            >>> C = cov.get_matrix(sigma=2.0, sigma0=3.0, scale=[1.0, 2.0])
            array([[13.        ,  3.61643745,  3.51285267,  3.47045163],
                   [ 3.61643745, 13.        ,  3.32078482,  3.14804532],
                   [ 3.51285267,  3.32078482, 13.        ,  3.53448631],
                   [ 3.47045163,  3.14804532,  3.53448631, 13.        ]])

        **Taking Derivatives:**

        Compute the second mixed derivative

        .. math::

            \\frac{\\partial^2}{\\partial \\alpha_1 \\partial \\alpha_2}
            \\boldsymbol{\\Sigma}(\\alpha_1, \\alpha_2 \\vert \\sigma,
            \\varsigma).

        .. code-block:: python
            :emphasize-lines: 2

            >>> # Compute second mixed derivative
            >>> C = cov.get_matrix(sigma=2.0, sigma0=3.0, scale=[1.0, 2.0],
            ...    derivative=[0, 1])
            array([[0.         0.04101073 0.01703885 0.0667311 ]
                   [0.04101073 0.         0.02500619 0.11654524]
                   [0.01703885 0.02500619 0.         0.00307613]
                   [0.0667311  0.11654524 0.00307613 0.        ]])

        Note that as mentioned in the above notes, its diagonal elements are
        zero.
        """

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if sigma < self.tol:

            if len(derivative) == 0:
                # Return scalar multiple of identity matrix
                S = sigma0**2 * self.mixed_cor.I
            else:
                # Return zero matrix
                n = self.mixed_cor.get_matrix_size()
                if self.cor.sparse:
                    S = scipy.sparse.csr_matrix((n, n))
                else:
                    S = numpy.zeros((n, n), dtype=float)

        else:

            eta = (sigma0 / sigma)**2
            Kn = self.mixed_cor.get_matrix(eta, scale, derivative)
            S = sigma**2 * Kn

        return S

    # =====
    # trace
    # =====

    def trace(
            self,
            sigma=None,
            sigma0=None,
            scale=None,
            p=1,
            derivative=[],
            imate_options={}):
        """
        Compute the trace of the positive powers of the covariance matrix or
        its derivatives.

        Parameters
        ----------

        sigma : float, default=None
            The hyperparameter :math:`\\sigma` of the covariance model where
            :math:`\\sigma^2` represents the variance of the correlated errors
            of the model. :math:`\\sigma` should be positive and cannot be
            `None`.

        sigma0 : float, default=None
            The hyperparameter :math:`\\varsigma` of the covariance model where
            :math:`\\varsigma^2` represents the variance of the input noise to
            of the model. :math:`\\sigma` should be positive and cannot be
            `None`.

        scale : float or array_like[float], default=None
            The scale hyperparameters
            :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_d)` in
            scales the distance between data points in :math:`\\mathbb{R}^d`.
            If an array of the size :math:`d` is given, each :math:`\\alpha_i`
            scales the distance in the :math:`i`-th dimension. If a scalar
            value :math:`\\alpha` is given, all dimensions are scaled
            isometrically. :math:`\\boldsymbol{\\alpha}` cannot be `None`.

        p : float, default=1
            The exponent :math:`p` of the covariance matrix
            :math:`\\boldsymbol{\\Sigma}^p` (see Notes below). The exponent
            should be non-negative real number. Note that if :math:`p \\neq 1`,
            the derivative order should be zero, meaning that no derivative
            should be taken by setting ``derivative=[]``.

            .. note::

                For :math:`\\boldsymbol{\\Sigma}^{-p}` with :math:`p > 0` see
                :func:`glearn.Covariance.trace`.

        derivative : list, default=[]
            Specifies a list of derivatives of covariance matrix with respect
            to the hyperparameters :math:`\\boldsymbol{\\alpha} = (\\alpha_1,
            \\dots, \\alpha_d)`. A list of the size :math:`q` with the
            components ``[i, j, ..., k]`` corresponds to take the derivative

            .. math::

                \\left. \\frac{\\partial^q}{\\partial \\alpha_{i+1} \\partial
                \\alpha_{j+1} \\dots \\partial \\alpha_{k+1}}
                \\boldsymbol{\\Sigma}^p(\\boldsymbol{\\alpha} \\vert
                \\sigma^2, \\varsigma^2) \\right|_{\\boldsymbol{\\alpha}}.

            .. note::

                The derivative with respect to each hyperparameter
                :math:`\\alpha_i` can be at most of the order two,
                :math:`\\partial^2 / \\partial \\alpha_i^2`. That is, each
                index in the ``derivative`` list can appear at most twice.
                For instance ``derivative=[1, 1]`` (second order derivative
                with respect to :math:`\\alpha_{2}`) is a valid input argument,
                how ever ``derivative=[1, 1, 1]`` (third order derivative) is
                an invalid input.

            .. note::
                When the derivative order is non-zero (meaning that
                ``derivative`` is not ``[]``), the exponent :math:`p` should
                be `1`.

        Returns
        -------

        S : numpy.ndarray
            An array of the size :math:`n \\times n` where :math:`n` is the
            size of the matrix.

        See Also
        --------

        glearn.Covariance.get_matrix
        glearn.Covariance.traceinv
        glearn.Covariance.logdet
        glearn.Covariance.solve
        glearn.Covariance.dot

        Notes
        -----

        This function computes

        .. math::

            \\mathrm{trace} \\left(
            \\frac{\\partial^q}{\\partial \\alpha_{i} \\partial
            \\alpha_{j} \\dots \\partial \\alpha_{k}}
            \\boldsymbol{\\Sigma}^p(\\boldsymbol{\\alpha} \\vert
            \\sigma, \\varsigma) \\right),

        where the covariance matrix :math:`\\boldsymbol{\\Sigma}` is defined by

        .. math::

            \\boldsymbol{\\Sigma}(\\boldsymbol{\\alpha}, \\sigma, \\varsigma) =
            \\sigma^2 \\mathbf{K}(\\boldsymbol{\\alpha}) + \\varsigma^2
            \\mathbf{I}.

        In the above, :math:`\\mathbf{I}` is the identity matrix and
        :math:`\\mathbf{K}` is the correlation matrix that depends on a set of
        scale hyperparameters :math:`\\boldsymbol{\\alpha}=(\\alpha_1, \\dots,
        \\alpha_d)`.

        **Derivatives:**

        Note that the indices in list ``derivative=[i, j, ..., k]`` are
        zero-indexed, meaning that the index ``i`` corresponds to take
        derivative with respect to the hyperparameter :math:`\\alpha_{i+1}`.
        For instance:

        * ``[]`` corresponds to no derivative.
        * ``[0]`` corresponds to :math:`\\partial / \\partial \\alpha_1` and
          ``[1]`` corresponds to :math:`\\partial / \\partial
          \\alpha_2`.
        * ``[0, 2]`` corresponds to :math:`\\partial^2 /
          \\partial \\alpha_1 \\partial \\alpha_3`.
        * ``[0, 0]`` corresponds to :math:`\\partial^2 /
          \\partial \\alpha_1^2`.
        * ``[0, 2, 2, 4]`` corresponds to :math:`\\partial^4 /
          \\partial \\alpha_1 \\partial \\alpha_{3}^2 \\partial \\alpha_5`.

        **Configuring Computation Settings:**

        This function passes the computation of trace to the function
        :func:`imate.trace`. To configure the latter function, create a
        dictionary of input arguments to this function and pass the dictionary
        with :func:`glearn.Covariance.set_imate_options`. See examples below
        for details.

        Examples
        --------

        **Basic Usage:**

        Create a sample dataset with four points in :math:`d=2` dimensional
        space. Then, compute the trace of
        :math:`\\boldsymbol{\\Sigma}^{2}(\\boldsymbol{\\alpha}, \\sigma,
        \\varsigma)` for :math:`\\boldsymbol{\\alpha} = (1, 2)`,
        :math:`\\sigma=2`, and :math:`\\varsigma=3`.

        .. code-block:: python
            :emphasize-lines: 10

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=4, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x)

            >>> # Compute trace
            >>> cov.trace(sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=2)
            817.7863657241508

        **Configure Computation:**

        The following example shows how to compute the trace of
        :math:`\\boldsymbol{\\Sigma}^{\\frac{3}{2}}(\\boldsymbol{\\alpha},
        \\sigma, \\varsigma)`.  Note that the exponent :math:`p` is not an
        integer. To compute the trace of non-integer exponents, the backend
        function :func:`imate.trace` should be configured to use either
        ``eigenvalue`` or ``slq`` methods. In the following example, the
        `eigenvalue` method is used.

        .. code-block:: python

            >>> # Check the default imate option
            >>> cov.get_imate_options()
            {
                'method': 'cholesky'
            }

        The above method (Cholesky) cannot compute the trace of non-integer
        exponents. In the following we change the method to eigenvalue method.

        .. code-block:: python
            :emphasize-lines: 8

            >>> # Change the default imate option
            >>> options = {
            ...    'method' : 'eigenvalue'
            ... }
            >>> cov.set_imate_options(options)

            >>> # Compute trace with non-integer exponent
            >>> cov.trace(sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=1.5)
            201.2755406790841

        **Taking Derivatives:**

        Compute the trace of the second mixed derivative

        .. math::

            \\frac{\\partial^2}{\\partial \\alpha_1 \\partial \\alpha_2}
            \\boldsymbol{\\Sigma}(\\alpha_1, \\alpha_2 \\vert \\sigma,
            \\varsigma).

        .. note::

            When taking the derivative, the exponent :math:`p` should be `1`.

        .. code-block:: python

            >>> # Compute second mixed derivative
            >>> cov.trace(sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=1,
            ...           derivative=[0, 1])
            0.0
        """

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if (p > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "p" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and p == 0:
            # Matrix is zero.
            trace_ = 0.0

        elif p == 0:
            # Matrix is identity.
            n = self.mixed_cor.get_matrix_size()
            trace_ = n

        if numpy.abs(sigma) < self.tol:

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                trace_ = 0.0
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                n = self.mixed_cor.get_matrix_size()
                trace_ = (sigma0**(2.0*p)) * n

        else:
            # Derivative eliminates sigma0^2 I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            trace_ = sigma**(2.0*p) * self.mixed_cor.trace(
                    eta, scale, p, derivative, imate_options)

        return trace_

    # ========
    # traceinv
    # ========

    def traceinv(
            self,
            sigma=None,
            sigma0=None,
            B=None,
            C=None,
            scale=None,
            p=1,
            derivative=[],
            imate_options={}):
        """
        Compute the trace of the negative powers of the covariance matrix or
        its derivatives.

        Parameters
        ----------

        sigma : float, default=None
            The hyperparameter :math:`\\sigma` of the covariance model where
            :math:`\\sigma^2` represents the variance of the correlated errors
            of the model. :math:`\\sigma` should be positive and cannot be
            `None`.

        sigma0 : float, default=None
            The hyperparameter :math:`\\varsigma` of the covariance model where
            :math:`\\varsigma^2` represents the variance of the input noise to
            of the model. :math:`\\sigma` should be positive and cannot be
            `None`.

        scale : float or array_like[float], default=None
            The scale hyperparameters
            :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_d)` in
            scales the distance between data points in :math:`\\mathbb{R}^d`.
            If an array of the size :math:`d` is given, each :math:`\\alpha_i`
            scales the distance in the :math:`i`-th dimension. If a scalar
            value :math:`\\alpha` is given, all dimensions are scaled
            isometrically. :math:`\\boldsymbol{\\alpha}` cannot be `None`.

        p : float, default=1
            The exponent :math:`p` of the covariance matrix
            :math:`\\boldsymbol{\\Sigma}^{-p}` (see Notes below). The exponent
            should be non-negative real number. Note that if :math:`p \\neq 1`,
            the derivative order should be zero, meaning that no derivative
            should be taken by setting ``derivative=[]``.

            .. note::

                For :math:`\\boldsymbol{\\Sigma}^p` with :math:`p > 0` see
                :func:`glearn.Covariance.trace`.

        derivative : list, default=[]
            Specifies a list of derivatives of covariance matrix with respect
            to the hyperparameters :math:`\\boldsymbol{\\alpha} = (\\alpha_1,
            \\dots, \\alpha_d)`. A list of the size :math:`q` with the
            components ``[i, j, ..., k]`` corresponds to take the derivative

            .. math::

                \\left. \\frac{\\partial^q}{\\partial \\alpha_{i+1} \\partial
                \\alpha_{j+1} \\dots \\partial \\alpha_{k+1}}
                \\boldsymbol{\\Sigma}^{-p}(\\boldsymbol{\\alpha} \\vert
                \\sigma^2, \\varsigma^2) \\right|_{\\boldsymbol{\\alpha}}.

            .. note::

                The derivative with respect to each hyperparameter
                :math:`\\alpha_i` can be at most of the order two,
                :math:`\\partial^2 / \\partial \\alpha_i^2`. That is, each
                index in the ``derivative`` list can appear at most twice.
                For instance ``derivative=[1, 1]`` (second order derivative
                with respect to :math:`\\alpha_{2}`) is a valid input argument,
                how ever ``derivative=[1, 1, 1]`` (third order derivative) is
                an invalid input.

            .. note::
                When the derivative order is non-zero (meaning that
                ``derivative`` is not ``[]``), the exponent :math:`p` should
                be `1`.

        Returns
        -------

        S : numpy.ndarray
            An array of the size :math:`n \\times n` where :math:`n` is the
            size of the matrix.

        See Also
        --------

        glearn.Covariance.get_matrix
        glearn.Covariance.trace
        glearn.Covariance.logdet
        glearn.Covariance.solve
        glearn.Covariance.dot

        Notes
        -----

        This function computes

        .. math::

            \\mathrm{trace} \\left(
            \\frac{\\partial^q}{\\partial \\alpha_{i} \\partial
            \\alpha_{j} \\dots \\partial \\alpha_{k}}
            \\boldsymbol{\\Sigma}^{-p}(\\boldsymbol{\\alpha} \\vert
            \\sigma, \\varsigma) \\right),

        where the covariance matrix :math:`\\boldsymbol{\\Sigma}` is defined by

        .. math::

            \\boldsymbol{\\Sigma}(\\boldsymbol{\\alpha}, \\sigma, \\varsigma) =
            \\sigma^2 \\mathbf{K}(\\boldsymbol{\\alpha}) + \\varsigma^2
            \\mathbf{I}.

        In the above, :math:`\\mathbf{I}` is the identity matrix and
        :math:`\\mathbf{K}` is the correlation matrix that depends on a set of
        scale hyperparameters :math:`\\boldsymbol{\\alpha}=(\\alpha_1, \\dots,
        \\alpha_d)`.

        **Derivatives:**

        Note that the indices in list ``derivative=[i, j, ..., k]`` are
        zero-indexed, meaning that the index ``i`` corresponds to take
        derivative with respect to the hyperparameter :math:`\\alpha_{i+1}`.
        For instance:

        * ``[]`` corresponds to no derivative.
        * ``[0]`` corresponds to :math:`\\partial / \\partial \\alpha_1` and
          ``[1]`` corresponds to :math:`\\partial / \\partial
          \\alpha_2`.
        * ``[0, 2]`` corresponds to :math:`\\partial^2 /
          \\partial \\alpha_1 \\partial \\alpha_3`.
        * ``[0, 0]`` corresponds to :math:`\\partial^2 /
          \\partial \\alpha_1^2`.
        * ``[0, 2, 2, 4]`` corresponds to :math:`\\partial^4 /
          \\partial \\alpha_1 \\partial \\alpha_{3}^2 \\partial \\alpha_5`.

        **Configuring Computation Settings:**

        This function passes the computation of trace to the function
        :func:`imate.traceinv`. To configure the latter function, create a
        dictionary of input arguments to this function and pass the dictionary
        with :func:`glearn.Covariance.set_imate_options`. See examples below
        for details.

        Examples
        --------

        **Basic Usage:**

        Create a sample dataset with four points in :math:`d=2` dimensional
        space. Then, compute the trace of
        :math:`\\boldsymbol{\\Sigma}^{-2}(\\boldsymbol{\\alpha}, \\sigma,
        \\varsigma)` for :math:`\\boldsymbol{\\alpha} = (1, 2)`,
        :math:`\\sigma=2`, and :math:`\\varsigma=3`.

        .. code-block:: python
            :emphasize-lines: 10

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=4, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x)

            >>> # Compute trace
            >>> cov.traceinv(sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=2)
            0.03470510350035956

        **Configure Computation:**

        The following example shows how to compute the trace of
        :math:`\\boldsymbol{\\Sigma}^{-\\frac{3}{2}}(\\boldsymbol{\\alpha},
        \\sigma, \\varsigma)`.  Note that the exponent :math:`p` is not an
        integer. To compute the trace of non-integer exponents, the backend
        function :func:`imate.traceinv` should be configured to use either
        ``eigenvalue`` or ``slq`` methods. In the following example, the
        `eigenvalue` method is used.

        .. code-block:: python

            >>> # Check the default imate option
            >>> cov.get_imate_options()
            {
                'method': 'cholesky'
            }

        The above method (Cholesky) cannot compute the trace of non-integer
        exponents. In the following we change the method to eigenvalue method.

        .. code-block:: python
            :emphasize-lines: 8

            >>> # Change the default imate option
            >>> options = {
            ...    'method' : 'eigenvalue'
            ... }
            >>> cov.set_imate_options(options)

            >>> # Compute trace with non-integer exponent
            >>> cov.traceinv(sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=1.5)
            0.11044285706736107

        **Taking Derivatives:**

        Compute the trace of the second mixed derivative

        .. math::

            \\frac{\\partial^2}{\\partial \\alpha_1 \\partial \\alpha_2}
            \\boldsymbol{\\Sigma}^{-1}(\\alpha_1, \\alpha_2 \\vert \\sigma,
            \\varsigma).

        .. note::

            When taking the derivative, the exponent :math:`p` should be `1`.

        .. code-block:: python

            >>> # Compute second mixed derivative
            >>> cov.traceinv(sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=1,
            ...              derivative=[0, 1])
            -866.1613714419709
        """

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if (B is None) and (C is not None):
            raise ValueError('When "C" is given, "B" should also be given.')

        if (p > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "p" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and p == 0:
            # Matrix is zero.
            traceinv_ = numpy.nan

        elif p == 0:
            # Matrix is identity, derivative is zero.
            if B is None:
                # B is identity
                n = self.mixed_cor.get_matrix_size()
                traceinv_ = n
            else:
                # B is not identity.
                if C is None:
                    traceinv_ = imate.trace(B, method='exact')
                else:
                    # C is not identity. Compute trace of C*B
                    if isspmatrix(C):
                        traceinv_ = numpy.sum(C.multiply(B.T).data)
                    else:
                        traceinv_ = numpy.sum(numpy.multiply(C, B.T))

        elif numpy.abs(sigma) < self.tol:

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                traceinv_ = numpy.nan
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                if B is None:
                    # B is identity
                    n = self.mixed_cor.get_matrix_size()
                    traceinv_ = n / (sigma0**(2.0*p))
                else:
                    # B is not identity
                    if C is None:
                        traceinv_ = imate.trace(B, method='exact') / \
                                (sigma0**(2.0*p))
                    else:
                        # C is not indentity. Compute trace of C*B devided by
                        # sigma0**4 (becase when we have C, there are to
                        # matrix A).
                        if isspmatrix(C):
                            traceinv_ = numpy.sum(C.multiply(B.T).data) / \
                                    (sigma0**(4.0*p))
                        else:
                            traceinv_ = numpy.sum(numpy.multiply(C, B.T)) / \
                                    (sigma0**(4.0*p))

        else:
            # Derivative eliminates sigma0^2*I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            traceinv_ = self.mixed_cor.traceinv(
                    eta, B, C, scale, p, derivative, imate_options)
            if C is None:
                traceinv_ /= sigma**(2.0*p)
            else:
                # When C is given, there are two A matrices (C*Ainv*B*Ainv)
                traceinv_ /= sigma**(4.0*p)

        return traceinv_

    # ======
    # logdet
    # ======

    def logdet(
            self,
            sigma=None,
            sigma0=None,
            scale=None,
            p=1,
            derivative=[],
            imate_options={}):
        """
        Compute the log-determinant of the powers of the covariance matrix or
        its derivatives.

        Parameters
        ----------

        sigma : float, default=None
            The hyperparameter :math:`\\sigma` of the covariance model where
            :math:`\\sigma^2` represents the variance of the correlated errors
            of the model. :math:`\\sigma` should be positive and cannot be
            `None`.

        sigma0 : float, default=None
            The hyperparameter :math:`\\varsigma` of the covariance model where
            :math:`\\varsigma^2` represents the variance of the input noise to
            of the model. :math:`\\sigma` should be positive and cannot be
            `None`.

        scale : float or array_like[float], default=None
            The scale hyperparameters
            :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_d)` in
            scales the distance between data points in :math:`\\mathbb{R}^d`.
            If an array of the size :math:`d` is given, each :math:`\\alpha_i`
            scales the distance in the :math:`i`-th dimension. If a scalar
            value :math:`\\alpha` is given, all dimensions are scaled
            isometrically. :math:`\\boldsymbol{\\alpha}` cannot be `None`.

        p : float, default=1
            The real exponent :math:`p` (negative or positive) of the
            covariance matrix :math:`\\boldsymbol{\\Sigma}^{p}` (see Notes
            below).

        derivative : list, default=[]
            Specifies a list of derivatives of covariance matrix with respect
            to the hyperparameters :math:`\\boldsymbol{\\alpha} = (\\alpha_1,
            \\dots, \\alpha_d)`. A list of the size :math:`q` with the
            components ``[i, j, ..., k]`` corresponds to take the derivative

            .. math::

                \\left. \\frac{\\partial^q}{\\partial \\alpha_{i+1} \\partial
                \\alpha_{j+1} \\dots \\partial \\alpha_{k+1}}
                \\boldsymbol{\\Sigma}^{p}(\\boldsymbol{\\alpha} \\vert
                \\sigma^2, \\varsigma^2) \\right|_{\\boldsymbol{\\alpha}}.

            .. note::

                The derivative with respect to each hyperparameter
                :math:`\\alpha_i` can be at most of the order two,
                :math:`\\partial^2 / \\partial \\alpha_i^2`. That is, each
                index in the ``derivative`` list can appear at most twice.
                For instance ``derivative=[1, 1]`` (second order derivative
                with respect to :math:`\\alpha_{2}`) is a valid input argument,
                how ever ``derivative=[1, 1, 1]`` (third order derivative) is
                an invalid input.

            .. note::
                When the derivative order is non-zero (meaning that
                ``derivative`` is not ``[]``), the exponent :math:`p` should
                be `1`.

        Returns
        -------

        S : numpy.ndarray
            An array of the size :math:`n \\times n` where :math:`n` is the
            size of the matrix.

        See Also
        --------

        glearn.Covariance.get_matrix
        glearn.Covariance.trace
        glearn.Covariance.traceinv
        glearn.Covariance.solve
        glearn.Covariance.dot

        Notes
        -----

        This function computes

        .. math::

            \\log \\det \\left(
            \\frac{\\partial^q}{\\partial \\alpha_{i} \\partial
            \\alpha_{j} \\dots \\partial \\alpha_{k}}
            \\boldsymbol{\\Sigma}^{p}(\\boldsymbol{\\alpha} \\vert
            \\sigma, \\varsigma) \\right),

        where the covariance matrix :math:`\\boldsymbol{\\Sigma}` is defined by

        .. math::

            \\boldsymbol{\\Sigma}(\\boldsymbol{\\alpha}, \\sigma, \\varsigma) =
            \\sigma^2 \\mathbf{K}(\\boldsymbol{\\alpha}) + \\varsigma^2
            \\mathbf{I}.

        In the above, :math:`\\mathbf{I}` is the identity matrix and
        :math:`\\mathbf{K}` is the correlation matrix that depends on a set of
        scale hyperparameters :math:`\\boldsymbol{\\alpha}=(\\alpha_1, \\dots,
        \\alpha_d)`.

        **Derivatives:**

        Note that the indices in list ``derivative=[i, j, ..., k]`` are
        zero-indexed, meaning that the index ``i`` corresponds to take
        derivative with respect to the hyperparameter :math:`\\alpha_{i+1}`.
        For instance:

        * ``[]`` corresponds to no derivative.
        * ``[0]`` corresponds to :math:`\\partial / \\partial \\alpha_1` and
          ``[1]`` corresponds to :math:`\\partial / \\partial
          \\alpha_2`.
        * ``[0, 2]`` corresponds to :math:`\\partial^2 /
          \\partial \\alpha_1 \\partial \\alpha_3`.
        * ``[0, 0]`` corresponds to :math:`\\partial^2 /
          \\partial \\alpha_1^2`.
        * ``[0, 2, 2, 4]`` corresponds to :math:`\\partial^4 /
          \\partial \\alpha_1 \\partial \\alpha_{3}^2 \\partial \\alpha_5`.

        **Configuring Computation Settings:**

        This function passes the computation of trace to the function
        :func:`imate.logdet`. To configure the latter function, create a
        dictionary of input arguments to this function and pass the dictionary
        with :func:`glearn.Covariance.set_imate_options`. See examples below
        for details.

        Examples
        --------

        **Basic Usage:**

        Create a sample dataset with four points in :math:`d=2` dimensional
        space. Then, compute the log-determinant of
        :math:`\\boldsymbol{\\Sigma}^{2}(\\boldsymbol{\\alpha}, \\sigma,
        \\varsigma)` for :math:`\\boldsymbol{\\alpha} = (1, 2)`,
        :math:`\\sigma=2`, and :math:`\\varsigma=3`.

        .. code-block:: python
            :emphasize-lines: 10

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=4, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x)

            >>> # Compute log-determinant
            >>> cov.logdet(sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=2)
            19.843781574740206

        **Configure Computation:**

        The following example shows how to compute the log-determinant of
        :math:`\\boldsymbol{\\Sigma}^{\\frac{3}{2}}(\\boldsymbol{\\alpha},
        \\sigma, \\varsigma)`. First, we check the default method:

        .. code-block:: python

            >>> # Check the default imate option
            >>> cov.get_imate_options()
            {
                'method': 'cholesky'
            }

        In the following, we change the method to eigenvalue method.

        .. code-block:: python
            :emphasize-lines: 8

            >>> # Change the default imate option
            >>> options = {
            ...    'method' : 'eigenvalue'
            ... }
            >>> cov.set_imate_options(options)

            >>> # Compute log-determinant with eigenvalue method
            >>> cov.logdet(sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=1.5)
            14.882836181055152

        **Taking Derivatives:**

        Compute the log-determinant of the second mixed derivative

        .. math::

            \\frac{\\partial^2}{\\partial \\alpha_2^2} \\boldsymbol{\\Sigma}
            (\\alpha_1, \\alpha_2 \\vert \\sigma, \\varsigma).

        .. code-block:: python

            >>> # Compute second mixed derivative
            >>> cov.logdet(sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=1,
            ...              derivative=[1, 1])
            8.095686613549319
        """

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if (p > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "p" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and p == 0:
            # Matrix is zero.
            logdet_ = -numpy.inf

        elif p == 0:
            # Matrix is identity.
            logdet_ = 0.0

        elif numpy.abs(sigma) < self.tol:

            n = self.mixed_cor.get_matrix_size()

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                logdet_ = -numpy.inf
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                logdet_ = (2.0*p*n) * numpy.log(sigma0)

        else:
            n = self.mixed_cor.get_matrix_size()

            # Derivative eliminates sigma0^2 I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            logdet_ = (2.0*p*n) * numpy.log(sigma) + \
                self.mixed_cor.logdet(eta, scale, p, derivative,
                                      imate_options)

        return logdet_

    # =====
    # solve
    # =====

    def solve(
            self,
            Y,
            sigma=None,
            sigma0=None,
            scale=None,
            p=1,
            derivative=[]):
        """
        Solve linear system involving the powers of covariance matrix or its
        derivatives.

        Parameters
        ----------

        Y : numpy.ndarray
            The right-hand side matrix of the linear system of equations. The
            size of this matrix is :math:`n \\times m` where :math:`n` is the
            size of the covariance matrix.

        sigma : float, default=None
            The hyperparameter :math:`\\sigma` of the covariance model where
            :math:`\\sigma^2` represents the variance of the correlated errors
            of the model. :math:`\\sigma` should be positive and cannot be
            `None`.

        sigma0 : float, default=None
            The hyperparameter :math:`\\varsigma` of the covariance model where
            :math:`\\varsigma^2` represents the variance of the input noise to
            of the model. :math:`\\sigma` should be positive and cannot be
            `None`.

        scale : float or array_like[float], default=None
            The scale hyperparameters
            :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_d)` in
            scales the distance between data points in :math:`\\mathbb{R}^d`.
            If an array of the size :math:`d` is given, each :math:`\\alpha_i`
            scales the distance in the :math:`i`-th dimension. If a scalar
            value :math:`\\alpha` is given, all dimensions are scaled
            isometrically. :math:`\\boldsymbol{\\alpha}` cannot be `None`.

        p : float, default=1
            The integer exponent :math:`p` (negative or positive) of the
            covariance matrix :math:`\\boldsymbol{\\Sigma}^{p}` (see Notes
            below).

        derivative : list, default=[]
            Specifies a list of derivatives of covariance matrix with respect
            to the hyperparameters :math:`\\boldsymbol{\\alpha} = (\\alpha_1,
            \\dots, \\alpha_d)`. A list of the size :math:`q` with the
            components ``[i, j, ..., k]`` corresponds to take the derivative

            .. math::

                \\left. \\frac{\\partial^q}{\\partial \\alpha_{i+1} \\partial
                \\alpha_{j+1} \\dots \\partial \\alpha_{k+1}}
                \\boldsymbol{\\Sigma}^{p}(\\boldsymbol{\\alpha} \\vert
                \\sigma^2, \\varsigma^2) \\right|_{\\boldsymbol{\\alpha}}.

            .. note::

                The derivative with respect to each hyperparameter
                :math:`\\alpha_i` can be at most of the order two,
                :math:`\\partial^2 / \\partial \\alpha_i^2`. That is, each
                index in the ``derivative`` list can appear at most twice.
                For instance ``derivative=[1, 1]`` (second order derivative
                with respect to :math:`\\alpha_{2}`) is a valid input argument,
                how ever ``derivative=[1, 1, 1]`` (third order derivative) is
                an invalid input.

            .. note::
                When the derivative order is non-zero (meaning that
                ``derivative`` is not ``[]``), the exponent :math:`p` should
                be `1`.

        Returns
        -------

        X : numpy.ndarray
            The solved array with the same size as of `Y`.

        See Also
        --------

        glearn.Covariance.get_matrix
        glearn.Covariance.dot

        Notes
        -----

        This function solves the linear system

        .. math::

            \\boldsymbol{\\Sigma}^{p}_{(i, j, \\dots, k)} \\mathbf{X} =
            \\mathbf{Y},

        where :math:`\\boldsymbol{\\Sigma}^{p}_{(i, j, \\dots, k)}` is defined
        as

        .. math::

            \\boldsymbol{\\Sigma}^{p}_{(i, j, \\dots, k)} =
            \\frac{\\partial^q}{\\partial \\alpha_{i} \\partial
            \\alpha_{j} \\dots \\partial \\alpha_{k}}
            \\boldsymbol{\\Sigma}^{p}(\\boldsymbol{\\alpha} \\vert
            \\sigma, \\varsigma).

        In the above, :math:`p` is the matrix exponent and :math:`q` is the
        order of derivation. Also, the covariance matrix
        :math:`\\boldsymbol{\\Sigma}` is defined by

        .. math::

            \\boldsymbol{\\Sigma}(\\boldsymbol{\\alpha}, \\sigma, \\varsigma) =
            \\sigma^2 \\mathbf{K}(\\boldsymbol{\\alpha}) + \\varsigma^2
            \\mathbf{I}.

        In the above, :math:`\\mathbf{I}` is the identity matrix and
        :math:`\\mathbf{K}` is the correlation matrix that depends on a set of
        scale hyperparameters :math:`\\boldsymbol{\\alpha}=(\\alpha_1, \\dots,
        \\alpha_d)`.

        **Derivatives:**

        Note that the indices in list ``derivative=[i, j, ..., k]`` are
        zero-indexed, meaning that the index ``i`` corresponds to take
        derivative with respect to the hyperparameter :math:`\\alpha_{i+1}`.
        For instance:

        * ``[]`` corresponds to no derivative.
        * ``[0]`` corresponds to :math:`\\partial / \\partial \\alpha_1` and
          ``[1]`` corresponds to :math:`\\partial / \\partial
          \\alpha_2`.
        * ``[0, 2]`` corresponds to :math:`\\partial^2 /
          \\partial \\alpha_1 \\partial \\alpha_3`.
        * ``[0, 0]`` corresponds to :math:`\\partial^2 /
          \\partial \\alpha_1^2`.
        * ``[0, 2, 2, 4]`` corresponds to :math:`\\partial^4 /
          \\partial \\alpha_1 \\partial \\alpha_{3}^2 \\partial \\alpha_5`.

        **Configuring Computation Settings:**

        This function passes the computation of the log-determinant to the
        function :func:`imate.logdet`. To configure the latter function, create
        a dictionary of input arguments to this function and pass the
        dictionary with :func:`glearn.Covariance.set_imate_options`. See
        examples below for details.

        Examples
        --------

        **Basic Usage:**

        Create a covariance matrix based on a set of sample data with four
        points in :math:`d=2` dimensional space.

        .. code-block:: python

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=4, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x)

        In the following, we create a sample right-hand side matrix
        :math:`\\mathbf{Y}` of the size :math:`n \\times 2`. The size of the
        covariance, :math:`n`, is also the same as the size of the number of
        points generated in the above. We solve the linear system

        .. math::

            \\boldsymbol{\\Sigma}^{2} \\mathbf{X} = \\mathbf{Y},

        for the hyperparameters :math:`\\sigma=2`, :math:`\\varsigma = 3`,
        and :math:`\\boldsymbol{\\alpha} = (1, 2)`.

        .. code-block:: python

            >>> import numpy
            >>> numpy.random.seed(0)
            >>> n = cov.get_size()
            >>> m = 2
            >>> Y = numpy.random.randn(n, m)

            >>> # Solve linear system.
            >>> cov.solve(Y, sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=2)
            array([[ 0.00661562,  0.00021078],
                   [-0.00194021,  0.02031901],
                   [ 0.00776767, -0.0137298 ],
                   [-0.00239916, -0.00373247]])

        **Taking Derivatives:**

        Solve the linear system involving the second mixed derivative

        .. math::

            \\boldsymbol{\\Sigma}_{(1, 2)} \\mathbf{X} = \\mathbf{Y},

        where here :math:`\\boldsymbol{\\Sigma}_{(1, 2)}` is

        .. math::

            \\boldsymbol{\\Sigma}_{(1, 2)} =
            \\frac{\\partial^2}{\\partial \\alpha_1 \\partial \\alpha_2}
            \\boldsymbol{\\Sigma} (\\boldsymbol{\\alpha} \\vert \\sigma,
            \\varsigma).

        .. code-block:: python

            >>> # Compute second mixed derivative
            >>> cov.solve(Y, sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=1,
            ...           derivative=[0, 1])
            array([[  -64.59508396,    46.20009763],
                   [   80.00325735,   -49.33936488],
                   [-1320.94755293,   817.88667267],
                   [  314.55331235,  -172.5169869 ]])
        """

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if (p > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "p" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and p == 0:
            # Matrix is zero, hence has no inverse.
            X = numpy.zeros_like(Y)
            X[:] = numpy.nan

        elif p == 0:
            # Matrix is identity.
            X = numpy.copy(Y)

        elif numpy.abs(sigma) < self.tol:

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                X = numpy.zeros_like(Y)
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                X = Y / (sigma0**(2*p))

        else:
            # Derivative eliminates sigma0^2 I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            X = self.mixed_cor.solve(
                    Y, eta, scale, p, derivative) / \
                (sigma**(2*p))

        return X

    # ===
    # dot
    # ===

    def dot(
            self,
            x,
            sigma=None,
            sigma0=None,
            scale=None,
            p=1,
            derivative=[]):
        """
        Matrix-vector or matrix-matrix multiplication involving the powers of
        covariance matrix or its derivatives.

        Parameters
        ----------

        X : numpy.ndarray
            The right-hand side array (either 1D or 2D array). The size of this
            array is :math:`n \\times m` where :math:`n` is the size of the
            covariance matrix.

        sigma : float, default=None
            The hyperparameter :math:`\\sigma` of the covariance model where
            :math:`\\sigma^2` represents the variance of the correlated errors
            of the model. :math:`\\sigma` should be positive and cannot be
            `None`.

        sigma0 : float, default=None
            The hyperparameter :math:`\\varsigma` of the covariance model where
            :math:`\\varsigma^2` represents the variance of the input noise to
            of the model. :math:`\\sigma` should be positive and cannot be
            `None`.

        scale : float or array_like[float], default=None
            The scale hyperparameters
            :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_d)` in
            scales the distance between data points in :math:`\\mathbb{R}^d`.
            If an array of the size :math:`d` is given, each :math:`\\alpha_i`
            scales the distance in the :math:`i`-th dimension. If a scalar
            value :math:`\\alpha` is given, all dimensions are scaled
            isometrically. :math:`\\boldsymbol{\\alpha}` cannot be `None`.

        p : float, default=1
            The integer exponent :math:`p` (negative or positive) of the
            covariance matrix :math:`\\boldsymbol{\\Sigma}^{p}` (see Notes
            below).

        derivative : list, default=[]
            Specifies a list of derivatives of covariance matrix with respect
            to the hyperparameters :math:`\\boldsymbol{\\alpha} = (\\alpha_1,
            \\dots, \\alpha_d)`. A list of the size :math:`q` with the
            components ``[i, j, ..., k]`` corresponds to take the derivative

            .. math::

                \\left. \\frac{\\partial^q}{\\partial \\alpha_{i+1} \\partial
                \\alpha_{j+1} \\dots \\partial \\alpha_{k+1}}
                \\boldsymbol{\\Sigma}^{p}(\\boldsymbol{\\alpha} \\vert
                \\sigma^2, \\varsigma^2) \\right|_{\\boldsymbol{\\alpha}}.

            .. note::

                The derivative with respect to each hyperparameter
                :math:`\\alpha_i` can be at most of the order two,
                :math:`\\partial^2 / \\partial \\alpha_i^2`. That is, each
                index in the ``derivative`` list can appear at most twice.
                For instance ``derivative=[1, 1]`` (second order derivative
                with respect to :math:`\\alpha_{2}`) is a valid input argument,
                how ever ``derivative=[1, 1, 1]`` (third order derivative) is
                an invalid input.

            .. note::
                When the derivative order is non-zero (meaning that
                ``derivative`` is not ``[]``), the exponent :math:`p` should
                be `1`.

        Returns
        -------

        Y : numpy.ndarray
            An array with the same size as of `X`.

        See Also
        --------

        glearn.Covariance.get_matrix
        glearn.Covariance.solve

        Notes
        -----

        This function performs the matrix multiplication

        .. math::

            \\mathbf{Y} =
            \\boldsymbol{\\Sigma}^{p}_{(i, j, \\dots, k)} \\mathbf{X},

        where :math:`\\boldsymbol{\\Sigma}^{p}_{(i, j, \\dots, k)}` is defined
        as

        .. math::

            \\boldsymbol{\\Sigma}^{p}_{(i, j, \\dots, k)} =
            \\frac{\\partial^q}{\\partial \\alpha_{i} \\partial
            \\alpha_{j} \\dots \\partial \\alpha_{k}}
            \\boldsymbol{\\Sigma}^{p}(\\boldsymbol{\\alpha} \\vert
            \\sigma, \\varsigma).

        In the above, :math:`p` is the matrix exponent and :math:`q` is the
        order of derivation. Also, the covariance matrix
        :math:`\\boldsymbol{\\Sigma}` is defined by

        .. math::

            \\boldsymbol{\\Sigma}(\\boldsymbol{\\alpha}, \\sigma, \\varsigma) =
            \\sigma^2 \\mathbf{K}(\\boldsymbol{\\alpha}) + \\varsigma^2
            \\mathbf{I}.

        In the above, :math:`\\mathbf{I}` is the identity matrix and
        :math:`\\mathbf{K}` is the correlation matrix that depends on a set of
        scale hyperparameters :math:`\\boldsymbol{\\alpha}=(\\alpha_1, \\dots,
        \\alpha_d)`.

        **Derivatives:**

        Note that the indices in list ``derivative=[i, j, ..., k]`` are
        zero-indexed, meaning that the index ``i`` corresponds to take
        derivative with respect to the hyperparameter :math:`\\alpha_{i+1}`.
        For instance:

        * ``[]`` corresponds to no derivative.
        * ``[0]`` corresponds to :math:`\\partial / \\partial \\alpha_1` and
          ``[1]`` corresponds to :math:`\\partial / \\partial
          \\alpha_2`.
        * ``[0, 2]`` corresponds to :math:`\\partial^2 /
          \\partial \\alpha_1 \\partial \\alpha_3`.
        * ``[0, 0]`` corresponds to :math:`\\partial^2 /
          \\partial \\alpha_1^2`.
        * ``[0, 2, 2, 4]`` corresponds to :math:`\\partial^4 /
          \\partial \\alpha_1 \\partial \\alpha_{3}^2 \\partial \\alpha_5`.

        Examples
        --------

        **Basic Usage:**

        Create a covariance matrix based on a set of sample data with four
        points in :math:`d=2` dimensional space.

        .. code-block:: python

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=4, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x)

        In the following, we create a sample right-hand side matrix
        :math:`\\mathbf{Y}` of the size :math:`n \\times 2`. The size of the
        covariance, :math:`n`, is also the same as the size of the number of
        points generated in the above. We perform the matrix-matrix
        multiplication:

        .. math::

            \\mathbf{Y} = \\boldsymbol{\\Sigma}^{2} \\mathbf{X},

        for the hyperparameters :math:`\\sigma=2`, :math:`\\varsigma = 3`,
        and :math:`\\boldsymbol{\\alpha} = (1, 2)`.

        .. code-block:: python

            >>> import numpy
            >>> numpy.random.seed(0)
            >>> n = cov.get_size()
            >>> m = 2
            >>> X = numpy.random.randn(n, m)

            >>> # Solve linear system.
            >>> cov.dot(X, sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=2)
            array([[13.57159509, 48.72998688],
                   [21.69530401, 87.37975138],
                   [46.31225613, 36.13137595],
                   [34.22135642, 43.08362434]])

        **Taking Derivatives:**

        Perform matrix-matrix multiplication involving the second mixed
        derivative

        .. math::

            \\boldsymbol{\\Sigma}_{(1, 2)} \\mathbf{X} = \\mathbf{Y},

        where here :math:`\\boldsymbol{\\Sigma}_{(1, 2)}` is

        .. math::

            \\boldsymbol{\\Sigma}_{(1, 2)} =
            \\frac{\\partial^2}{\\partial \\alpha_1 \\partial \\alpha_2}
            \\boldsymbol{\\Sigma} (\\boldsymbol{\\alpha} \\vert \\sigma,
            \\varsigma).

        .. code-block:: python

            >>> # Compute second mixed derivative
            >>> cov.dot(Y, sigma=2.0, sigma0=3.0, scale=[1.0, 2.0], p=1,
            ...         derivative=[0, 1])
            array([[ 0.13536024,  0.06514873],
                   [ 0.22977385, -0.02566722],
                   [ 0.05745453,  0.06238882],
                   [ 0.23752925,  0.28486212]])
        """

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if (p > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "p" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif p == 0 and len(derivative) > 0:
            # Matrix is zero.
            y = numpy.zeros_like(x)

        elif p == 0:
            # Matrix is identity.
            y = x.copy()

        elif numpy.abs(sigma) < self.tol:

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                y = numpy.zeros_like(x)
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                y = sigma0**(2.0*p) * x

        else:
            # Derivative eliminates sigma0^2 I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            y = (sigma**(2.0*p)) * \
                self.mixed_cor.dot(x, eta, scale, p, derivative)

        return y

    # ===============
    # auto covariance
    # ===============

    def auto_covariance(self, training_points):
        """
        Compute the auto-covariance between a set of test points.

        Parameters
        ----------

        training_points : numpy.ndarray
            An array of the size :math:`n^{\\ast} \\times d` representing the
            coordinates of :math:`n^{\\ast}` test points. Each row of the array
            is the coordinates of a point
            :math:`\\boldsymbol{x} = (x_1, \\dots, x_d)`.

        Returns
        -------

        S_star_star : numpy.ndarray
            The covariance array :math:`\\boldsymbol{\\Sigma}^{\\ast \\ast}` of
            the size :math:`n^{\\ast} \\times n^{\\ast}`.

        See Also
        --------

        glearn.Covariance.get_matrix
        glearn.Covariance.auto_covariance

        Notes
        -----

        **Auto-Covariance:**

        Given a set of test points :math:`\\{ \\boldsymbol{x}^{\\ast}_i
        \\}_{i=1}^{n^{\\ast}}`, this function generates the :math:`n^{\\ast}
        \\times n^{\\ast}` auto-covariance
        :math:`\\boldsymbol{\\Sigma}^{\\ast \\ast}` where each element
        :math:`\\Sigma^{\\ast \\ast}_{ij}` of the matrix is the covariance
        between the points in the :math:`i`-th and :math:`j`-th test point,
        namely,

        .. math::

            \\Sigma^{\\ast \\ast}_{ij} = \\mathrm{cov}(
            \\boldsymbol{x}^{\\ast}_i, \\boldsymbol{x}^{\\ast}_j).

        **Specifying Hyperparameters:**

        The auto-covariance matrix :math:`\\boldsymbol{\\Sigma}^{\\ast \\ast}`
        depends on a set of hyperparameters as it is defined by

        .. math::

            \\boldsymbol{\\Sigma}^{\\ast \\ast}(\\boldsymbol{\\alpha},
            \\sigma, \\varsigma) =
            \\sigma^2 \\mathbf{K}^{\\ast \\ast}(\\boldsymbol{\\alpha}) +
            \\varsigma^2 \\mathbf{I}.

        In the above, :math:`\\mathbf{I}` is the identity matrix and
        :math:`\\mathbf{K}^{\\ast \\ast}` is the auto-correlation matrix that
        depends on a set of scale hyperparameters
        :math:`\\boldsymbol{\\alpha}=(\\alpha_1, \\dots, \\alpha_d)`.

        .. note::

            Before using :func:`glearn.Covariance.auto_covariance`, the
            hyperparameters :math:`\\sigma`, :math:`\\varsigma`, and
            :math:`\\boldsymbol{\\alpha}` of the covariance object should be
            defined. These hyperparameters can be either defined at the time of
            instantiation of :class:`glearn.Covariance`, or to be set by

            * :func:`glearn.Covariance.set_sigmas` to set :math:`\\sigma` and
              :math:`\\varsigma`.
            * :func:`glearn.Covariance.set_scale` to set
              :math:`\\boldsymbol{\\alpha}`.

        **Summary of Covariance Functions:**

        Suppose :math:`\\{ \\boldsymbol{x}_i \\}_{i=1}^{n}` and
        :math:`\\{ \\boldsymbol{x}^{\\ast}_i \\}_{i=1}^{n^{\\ast}}` are
        respectively training and test points. Three covariance matrices
        can be generated:

        * :func:`glearn.Covariance.get_matrix` returns the auto-covariance
          between training points by the :math:`n \\times n` matrix
          :math:`\\boldsymbol{\\Sigma}` with the components

          .. math::

            \\Sigma_{ij} = \\mathrm{cov}(\\boldsymbol{x}_i,
            \\boldsymbol{x}_j).

        * :func:`glearn.Covariance.cross_covariance` returns the
          cross-covariance between the training points and test points by the
          :math:`n \\times n^{\\ast}` matrix
          :math:`\\boldsymbol{\\Sigma}^{\\ast}` with the components

          .. math::

            \\Sigma_{ij}^{\\ast} = \\mathrm{cov}(\\boldsymbol{x}_i,
            \\boldsymbol{x}^{\\ast}_j).

        * :func:`glearn.Covariance.auto_covariance` returns the
          cross-covariance between the test points by the
          :math:`n^{\\ast} \\times n^{\\ast}` matrix
          :math:`\\boldsymbol{\\Sigma}^{\\ast \\ast}` with the components

          .. math::

            \\Sigma_{ij}^{\\ast \\ast} = \\mathrm{cov}(
            \\boldsymbol{x}^{\\ast}_i, \\boldsymbol{x}^{\\ast}_j).

        Examples
        --------

        Create a covariance matrix based on a set of sample data with four
        points in :math:`d=2` dimensional space.

        .. code-block:: python

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=4, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x, sigma=2.0, sigma0=3.0, scale=[1.0, 2.0])

        Now, create a set of test points :math:`\\boldsymbol{x}^{\\ast}`, and
        compute the auto-covariance between the test points.

        .. code-block:: python

            >>> # Generate a random set of points
            >>> x_star = generate_points(num_points=4, dimension=2, seed=42)

            >>> # Auto-covariance between test points
            >>> cov.auto_covariance(x_star)
            array([[4.        , 2.68545065, 2.54164549, 2.9067261 ],
                   [2.68545065, 4.        , 2.15816164, 2.01221865],
                   [2.54164549, 2.15816164, 4.        , 2.76750357],
                   [2.9067261 , 2.01221865, 2.76750357, 4.        ]])
        """

        if self.sigma is None:
            raise RuntimeError('"sigma" cannot be None to create auto ' +
                               'covariance.')

        auto_cor = self.cor.auto_correlation(training_points)
        auto_cov = (self.sigma**2) * auto_cor

        return auto_cov

    # ================
    # cross covariance
    # ================

    def cross_covariance(self, test_points):
        """
        Compute the cross-covariance between training and test points.

        Parameters
        ----------

        training_points : numpy.ndarray
            An array of the size :math:`n^{\\ast} \\times d` representing the
            coordinates of :math:`n^{\\ast}` test points. Each row of the array
            is the coordinates of a point
            :math:`\\boldsymbol{x} = (x_1, \\dots, x_d)`.

        Returns
        -------

        S_star_star : numpy.ndarray
            The covariance array :math:`\\boldsymbol{\\Sigma}^{\\ast}` of the
            size :math:`n \\times n^{\\ast}` where :math:`n` and
            :math:`n^{\\ast}` are respectively the number of training and test
            points.

        See Also
        --------

        glearn.Covariance.get_matrix
        glearn.Covariance.cross_covariance

        Notes
        -----

        **Cross-Covariance:**

        Suppose the training points :math:`\\{ \\boldsymbol{x}_i
        \\}_{i=1}^{n}` that were given at the time of creation of the
        covariance object. Given a set of test points
        :math:`\\{ \\boldsymbol{x}^{\\ast}_i \\}_{i=1}^{n^{\\ast}}`,
        this function generates the :math:`n \\times n^{\\ast}`
        cross-covariance :math:`\\boldsymbol{\\Sigma}^{\\ast \\ast}`  where
        each element :math:`\\Sigma^{\\ast}_{ij}` of the matrix is the
        cross covariance between the points in the :math:`i`-th training point
        and :math:`j`-th test point, namely,

        .. math::

            \\Sigma^{\\ast}_{ij} = \\mathrm{cov}(
            \\boldsymbol{x}_i, \\boldsymbol{x}^{\\ast}_j).

        **Specifying Hyperparameters:**

        The cross-covariance matrix :math:`\\boldsymbol{\\Sigma}^{\\ast}`
        depends on a set of hyperparameters as it is defined by

        .. math::

            \\boldsymbol{\\Sigma}^{\\ast}(\\boldsymbol{\\alpha},
            \\sigma, \\varsigma) =
            \\sigma^2 \\mathbf{K}^{\\ast}(\\boldsymbol{\\alpha}) +
            \\varsigma^2 \\mathbf{I}.

        In the above, :math:`\\mathbf{I}` is the identity matrix and
        :math:`\\mathbf{K}^{\\ast}` is the cross-correlation matrix that
        depends on a set of scale hyperparameters
        :math:`\\boldsymbol{\\alpha}=(\\alpha_1, \\dots, \\alpha_d)`.

        .. note::

            Before using :func:`glearn.Covariance.cross_covariance`, the
            hyperparameters :math:`\\sigma`, :math:`\\varsigma`, and
            :math:`\\boldsymbol{\\alpha}` of the covariance object should be
            defined. These hyperparameters can be either defined at the time of
            instantiation of :class:`glearn.Covariance`, or to be set by

            * :func:`glearn.Covariance.set_sigmas` to set :math:`\\sigma` and
              :math:`\\varsigma`.
            * :func:`glearn.Covariance.set_scale` to set
              :math:`\\boldsymbol{\\alpha}`.

        **Summary of Covariance Functions:**

        Suppose :math:`\\{ \\boldsymbol{x}_i \\}_{i=1}^{n}` and
        :math:`\\{ \\boldsymbol{x}^{\\ast}_i \\}_{i=1}^{n^{\\ast}}` are
        respectively training and test points. Three covariance matrices
        can be generated:

        * :func:`glearn.Covariance.get_matrix` returns the auto-covariance
          between training points by the :math:`n \\times n` matrix
          :math:`\\boldsymbol{\\Sigma}` with the components

          .. math::

            \\Sigma_{ij} = \\mathrm{cov}(\\boldsymbol{x}_i,
            \\boldsymbol{x}_j).

        * :func:`glearn.Covariance.cross_covariance` returns the
          cross-covariance between the training points and test points by the
          :math:`n \\times n^{\\ast}` matrix
          :math:`\\boldsymbol{\\Sigma}^{\\ast}` with the components

          .. math::

            \\Sigma_{ij}^{\\ast} = \\mathrm{cov}(\\boldsymbol{x}_i,
            \\boldsymbol{x}^{\\ast}_j).

        * :func:`glearn.Covariance.auto_covariance` returns the
          cross-covariance between the test points by the
          :math:`n^{\\ast} \\times n^{\\ast}` matrix
          :math:`\\boldsymbol{\\Sigma}^{\\ast \\ast}` with the components

          .. math::

            \\Sigma_{ij}^{\\ast \\ast} = \\mathrm{cov}(
            \\boldsymbol{x}^{\\ast}_i, \\boldsymbol{x}^{\\ast}_j).

        Examples
        --------

        Create a covariance matrix based on a set of sample data with four
        points in :math:`d=2` dimensional space.

        .. code-block:: python

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=4, dimension=2)

            >>> # Create a covariance object
            >>> from glearn import Covariance
            >>> cov = Covariance(x, sigma=2.0, sigma0=3.0, scale=[1.0, 2.0])

        Now, create a set of test points :math:`\\boldsymbol{x}^{\\ast}`, and
        compute the auto-covariance between the test points.

        .. code-block:: python

            >>> # Generate a random set of points
            >>> x_star = generate_points(num_points=2, dimension=2, seed=42)

            >>> # Auto-covariance between test points
            >>> cov.cross_covariance(x_star)
            array([[3.24126331, 3.30048921],
                   [2.94735574, 3.50537082],
                   [3.40813768, 2.93601147],
                   [3.7310863 , 2.87895123]])
        """

        if self.sigma is None:
            raise RuntimeError('"sigma" cannot be None to create cross ' +
                               'covariance.')

        cross_cor = self.cor.cross_correlation(test_points)
        cross_cov = (self.sigma**2) * cross_cor

        return cross_cov
