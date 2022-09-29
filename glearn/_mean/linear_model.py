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
from detkit import orthogonalize

__all__ = ['LinearModel']


# ============
# linear model
# ============

class LinearModel(object):
    """
    Linear model for the mean function of regression.

    For the regression problem
    :math:`y=\\mu(\\boldsymbol{x})+\\delta(\\boldsymbol{x})` where
    :math:`\\mu` and :math:`\\delta` are the mean and miss-fit of the
    regression model respectively, this class implements the linear model
    :math:`\\mu(\\boldsymbol{x})=
    \\boldsymbol{\\phi}(\\boldsymbol{x})^{\\intercal} \\boldsymbol{\\beta}`,
    where:

    * The basis functions :math:`\\boldsymbol{\\phi}` can be specified
      by polynomial, trigonometric, hyperbolic, or user-defined arbitrary
      functions.
    * The uniform or multi-dimensional normal prior distribution can be
      assigned to the vector of regression coefficients
      :math:`\\boldsymbol{\\beta}\\sim \\mathcal{N}(\\boldsymbol{b},\\sigma^2
      \\mathbf{B})`.

    Parameters
    ----------

    x : numpy.ndarray
        A 2D array of data points where each row of the array is the coordinate
        of a point :math:`\\boldsymbol{x} = (x_1, \\dots, x_d)`. The
        array size is :math:`n \\times d` where :math:`n` is the number of the
        points and :math:`d` is the dimension of the space of points.

    polynomial_degree : int, default=0
        Degree :math:`p` of polynomial basis functions
        :math:`\\boldsymbol{\\phi}_p(\\boldsymbol{x})`, which defines the
        monomial functions of order less than or equal to :math:`p` on the
        variables :math:`\\boldsymbol{x} = (x_1, \\dots, x_d)`. The total
        number of monomials is denoted by :math:`m_p`.

    trigonometric_coeff : float or array_like[float], default=None
        The coefficients :math:`t_i, i=1,\\dots, m_t`, of the trigonometric
        basis functions :math:`\\boldsymbol{\\phi}_t(\\boldsymbol{x})` (see
        Notes below). If `None`, no trigonometric basis function is generated.
        The total number of trigonometric basis functions is :math:`2 m_t`.

    hyperbolic_coeff : float or array_like[float], default=None
        The coefficients :math:`h_i, i=1, \\dots, m_h`, of the hyperbolic
        basis functions :math:`\\boldsymbol{\\phi}_h(\\boldsymbol{x})` (see
        Notes below). If `None`, no hyperbolic basis function is generated. The
        total number of hyperbolic basis functions is :math:`2 m_h`.

    func : callable, default=None
        A callable function to define arbitrary basis functions
        :math:`\\boldsymbol{\\phi}(\\boldsymbol{x})` where it accepts an input
        point :math:`\\boldsymbol{x}=(x_1, \\dots, m_d)` and returns the
        array of :math:`m_f` basis functions :math:`\\boldsymbol{\\phi} = (
        \\phi_1, \\dots, \\phi_{m_f})`. If `None`, no arbitrary basis function
        is generated. The total number of user-defined basis functions is
        denoted by :math:`m_f`.

    orthonormalize : bool, default=False
        If `True`, the design matrix :math:`\\mathbf{X}` of the basis functions
        will be orthonormalized. The orthonormalization is applied on all
        :math:`m` basis functions, including polynomial, trigonometric,
        hyperbolic, and user-defined basis functions.

    b : numpy.array, default=None,
        The variable :math:`\\boldsymbol{b}` which is the mean of the prior
        distribution for the variable :math:`\\boldsymbol{\\beta} \\sim
        \\mathcal{N}(\\boldsymbol{b}, \\sigma^2 \\mathbf{B})`. The variable
        ``b`` should be given as a 1D array of the size :math:`m = m_p+2m_t+
        2m_h+m_f`. If `None`, uniform prior is assumed for the variable
        :math:`\\boldsymbol{\\beta}`.

    B : numpy.array, default=None,
        The matrix :math:`\\mathbf{B}` which makes the covariance of the prior
        distribution for the variable :math:`\\boldsymbol{\\beta} \\sim
        \\mathcal{N}(\\boldsymbol{b}, \\sigma^2 \\mathbf{B})`. The matrix
        ``B`` should be symmetric and positive semi-definite with the size
        :math:`m \\times m` where :math:`m = m_p+2m_t+2m_h+m_f`.

        * If ``B`` is not `None`, the vector ``b`` also cannot be `None`.
        * If ``B`` is `None`, it is assumed that
          :math:`\\mathbf{B}^{-1} \\to \\mathbf{0}`, which implies the prior on
          :math:`\\boldsymbol{\\beta}` is the uniform distribution. In this
          case, the argument ``b`` (if given) is ignored.

    Attributes
    ----------

    points : numpy.ndarray
        The same as input variable ``x``.

    X : numpy.ndarray
        The generated design matrix.

    Binv : numpy.ndarray, default=None
        Inverse of the matrix ``B`` (if not `None`).

    beta : numpy.array
        The mean of the coefficient :math:`\\boldsymbol{\\beta}`. This
        coefficient is computed for a given data.

    C : numpy.ndarray
        The posterior covariance of coefficient :math:`\\boldsymbol{\\beta}`.
        This matrix computed for a given data.

    Methods
    -------

    generate_design_matrix
    update_hyperparam

    See Also
    --------

    glearn.Covariance
    glearn.GaussianProcess

    Notes
    -----

    **Regression Model:**

    A regression model to fit the data :math:`y = f(\\boldsymbol{x})`
    for the points :math:`\\boldsymbol{x} \\in \\mathcal{D} \\in \\mathbb{R}^d`
    and data :math:`y \\in \\mathbb{R}` is

    .. math::

        f(\\boldsymbol{x}) = \\mu(\\boldsymbol{x}) + \\delta(\\boldsymbol{x}),

    where :math:`\\mu` is a deterministic function representing the features
    of the data and :math:`\\delta` is a stochastic function representing both
    the uncertainty of the data and miss-fit of the regression model.

    **Linear Regression Model:**

    This class represents a linear regression model, where it is assumed that

    .. math::

        \\mu(\\boldsymbol{x}) =
        \\boldsymbol{\\phi}(\\boldsymbol{x})^{\\intercal} \\boldsymbol{\\beta},

    where :math:`\\boldsymbol{\\phi} = (\\phi_1, \\dots, \\phi_m):
    \\mathcal{D} \\to \\mathbb{R}^m` is a vector basis function and
    :math:`\\boldsymbol{\\beta} = (\\beta_1, \\dots, \\beta_m)` is a vector of
    the unknown coefficients of the linear model. The basis functions can be a
    combination of the following functions:

    .. math::

        \\boldsymbol{\\phi}(\\boldsymbol{x}) = \\left(
        \\boldsymbol{\\phi}_p(\\boldsymbol{x}),
        \\boldsymbol{\\phi}_t(\\boldsymbol{x}),
        \\boldsymbol{\\phi}_h(\\boldsymbol{x}),
        \\boldsymbol{\\phi}_f(\\boldsymbol{x}) \\right),

    consisting of

    * Polynomial basis functions :math:`\\boldsymbol{\\phi}_p: \\mathcal{D}
      \\to \\mathbb{R}^{m_p}`.
    * Trigonometric basis functions :math:`\\boldsymbol{\\phi}_t: \\mathcal{D}
      \\to \\mathbb{R}^{2m_t}`.
    * Hyperbolic basis functions :math:`\\boldsymbol{\\phi}_h: \\mathcal{D}
      \\to \\mathbb{R}^{2m_h}`.
    * User-defined basis functions :math:`\\boldsymbol{\\phi}_f: \\mathcal{D}
      \\to \\mathbb{R}^{m_f}`.

    The total number of the basis functions, :math:`m`, is

    .. math::

        m = m_p + 2m_t + 2m_h + m_f.

    Each of the above functions are described below.

    * **Polynomial Basis Functions:** A polynomial basis function of order
      :math:`p` (set by ``polynomial_order``) is a tuple of all monomials up to
      the order less than or equal to :math:`p` from the combination of the
      components of the variable :math:`\\boldsymbol{x} = (x_1, \\dots, x_d)`.
      For instance, if :math:`d = 2` where
      :math:`\\boldsymbol{x} = (x_1, x_2)`, a polynomial basis of order 3 is

      .. math::

        \\boldsymbol{\\phi}_p(x_1, x_2) = ( \\
            1, \\
            x_1, x_2, \\
            x_1^2, x_1 x_2, x_2^2, \\
            x_1^3, x_1^2 x_2, x_1 x_2^2, x_2^3).

      The size of the tuple :math:`\\boldsymbol{\\phi}_p` is denoted by
      :math:`m_p`, for instance, in the above, :math:`m_p = 10`.

    * **Trigonometric Basis Function:** Given the coefficients
      :math:`(t_1, \\dots, t_{m_t})` which can be set by
      ``trigonometric_coeff`` as a list or numpy array, the trigonometric basis
      functions :math:`\\boldsymbol{\\phi}_t` is defined by

      .. math::

        \\boldsymbol{\\phi}_t(\\boldsymbol{x}) =
        (\\boldsymbol{\\phi}_s(\\boldsymbol{x}),
        \\boldsymbol{\\phi}_c(\\boldsymbol{x})),

      where

      .. math::

        \\begin{align}
            \\boldsymbol{\\phi}_s(\\boldsymbol{x}) = (
            & \\sin(t_1 x_1), \\dots, \\sin(t_1 x_d), \\\\
            & \\dots, \\\\
            & \\sin(t_i x_1), \\dots, \\sin(t_i x_d), \\\\
            & \\dots, \\\\
            & \\sin(t_{m_t} x_1), \\dots, \\sin(t_{m_t} x_d)).
        \\end{align}

      and

      .. math::

        \\begin{align}
            \\boldsymbol{\\phi}_c(\\boldsymbol{x}) = (
            & \\cos(t_1 x_1), \\dots, \\cos(t_1 x_d), \\\\
            & \\dots, \\\\
            & \\cos(t_i x_1), \\dots, \\cos(t_i x_d), \\\\
            & \\dots, \\\\
            & \\cos(t_{m_t} x_1), \\dots, \\cos(t_{m_t} x_d)).
        \\end{align}

    * **Hyperbolic Basis Function:** Given the coefficients
      :math:`(h_1, \\dots, h_{m_h})` which can be set by
      ``hyperbolic_coeff`` as a list or numpy array, the hyperbolic basis
      functions :math:`\\boldsymbol{\\phi}_h` is defined by

      .. math::

        \\boldsymbol{\\phi}_h(\\boldsymbol{x}) =
        (\\boldsymbol{\\phi}_{sh}(\\boldsymbol{x}),
        \\boldsymbol{\\phi}_{ch}(\\boldsymbol{x})),

      where

      .. math::

        \\begin{align}
            \\boldsymbol{\\phi}_{sh}(\\boldsymbol{x}) = (
            & \\sinh(h_1 x_1), \\dots, \\sinh(h_1 x_d), \\\\
            & \\dots, \\\\
            & \\sinh(h_i x_1), \\dots, \\sinh(h_i x_d), \\\\
            & \\dots, \\\\
            & \\sinh(h_{m_h} x_1), \\dots, \\sinh(h_{m_h} x_d)).
        \\end{align}

      and

      .. math::

        \\begin{align}
            \\boldsymbol{\\phi}_{ch}(\\boldsymbol{x}) = (
            & \\cosh(h_1 x_1), \\dots, \\cosh(h_1 x_d), \\\\
            & \\dots, \\\\
            & \\cosh(h_i x_1), \\dots, \\cosh(h_i x_d), \\\\
            & \\dots, \\\\
            & \\cosh(h_{m_h} x_1), \\dots, \\cosh(h_{m_h} x_d)).
        \\end{align}

    * **User-Defined Basis Functions:** Given the function
      :math:`\\boldsymbol{\\phi}_f(\\boldsymbol{x})` which can be set by the
      argument ``func``, the custom basis function are generated by

      .. math::

        \\boldsymbol{\\phi}_f(\\boldsymbol{x}) =
        \\left( \\boldsymbol{\\phi}_{f, 1}(\\boldsymbol{x}),
        \\dots,
        \\boldsymbol{\\phi}_{f, m_f}(\\boldsymbol{x}) \\right).

    **Design Matrix:**

    The design matrix :math:`\\mathbf{X}` of the size :math:`n \\times m` is
    generated  from :math:`n` points :math:`\\boldsymbol{x}` and :math:`m`
    basis functions with the components :math:`X_{ij}` as

    .. math::

        X_{ij} = \\phi_j(\\boldsymbol{x}_i), \\quad i = 1, \\dots, n, \\quad
        j = 1, \\dots, m.

    If ``orthonormalize`` is `True`, the matrix :math:`\\mathbf{X}` is
    orthonormalized such that
    :math:`\\mathbf{X}^{\\intercal} \\mathbf{X} = \\mathbf{I}` where
    :math:`\\mathbf{I}` is the identity matrix.

    .. note::

        The design matrix on *data points* is automatically generated during
        the internal training process and can be accessed by ``X`` attribute.
        To generate the design matrix on arbitrary *test points*, call
        :meth:`glearn.LinearModel.generate_design_matrix` function.

    **Prior of Unknown Coefficients:**

    The coefficients :math:`\\boldsymbol{\\beta}` of the size :math:`m` is
    unknown at the time of instantiation of :class:`glearn.LinearModel` class.
    However, its prior distribution can be specified as

    .. math::

        \\boldsymbol{\\beta} \\sim \\mathcal{N}(\\boldsymbol{b}, \\sigma^2
        \\mathbf{B}),

    where the hyperparameter :math:`\\sigma` is to be found during the training
    process, and the hyperparameters :math:`\\boldsymbol{b}` and
    :math:`\\mathbf{B}` can be specified by the arguments ``b`` and ``B``
    respectively.

    **Posterior of Estimated Coefficients:**

    Once the model has been trained by :class:`glearn.GaussianProcess`, the
    hyperparameters of the posterior distribution

    .. math::

        \\boldsymbol{\\beta} \\sim \\mathcal{N}(\\bar{\\boldsymbol{\\beta}},
        \\mathbf{C}),

    can be accessed by these attributes:

    * :math:`\\bar{\\boldsymbol{\\beta}}` can be accessed by
      ``LinearModel.beta``.
    * :math:`\\mathbf{C}` can be accessed by ``LinearModel.C``.

    .. note::

        The hyperparameters of the posterior distribution of
        :math:`\\boldsymbol{\\beta}` is automatically computed **after**
        the training process. However, to manually update the hyperparameters
        of the posterior distribution after the training process, call
        :meth:`glearn.LinearModel.update_hyperparam` function.

    Examples
    --------

    **Create Polynomial, Trigonometric, and Hyperbolic Basis Functions:**

    First, create a set of 50 random points in the interval :math:`[0, 1]`.

    .. code-block:: python

        >>> # Generate a set of points
        >>> from glearn.sample_data import generate_points
        >>> x = generate_points(num_points=30, dimension=1)

    Define a prior for :math:`\\boldsymbol{\\beta} \\vert \\sigma^2 \\sim
    \\mathcal{N}(\\boldsymbol{b}, \\sigma^2 \\mathbf{B})` by the mean vector
    :math:`\\boldsymbol{b}` and covariance matrix :math:`\\mathbf{B}`.

    .. code-block:: python

        >>> import numpy
        >>> m = 10
        >>> b = numpy.zeros((m, ))

        >>> # Generate a random matrix B for covariance of beta.
        >>> numpy.random.seed(0)
        >>> B = numpy.random.rand(m, m)

        >>> # Making sure the covariance matrix B positive-semidefinite
        >>> B = B.T @ B

    **Create User-Defined Basis Functions:**

    Define the function :math:`\\boldsymbol{\\phi}` as follows:

    .. code-block:: python

        >>> # Create a function returning the Legendre Polynomials
        >>> import numpy
        >>> def func(x):
        ...     phi_1 = 0.5 * (3.0*x**2 - 1)
        ...     phi_2 = 0.5 * (5.0*x**3 - 3.0*x)
        ...     phi = numpy.array([phi_1, phi_2])
        ...     return phi

    **Create Linear Model:**

    Create a linear model with first order polynomial basis functions, both
    trigonometric and hyperbolic functions, and the user-defined functions
    created in the above.

    .. code-block:: python
        :emphasize-lines: 3, 4, 5

        >>> # Create basis functions
        >>> from glearn import LinearModel
        >>> f = LinearModel(x, polynomial_degree=1,
        ...                 trigonometric_coeff=[1.0, 2.0],
        ...                 hyperbolic_coeff=[3.0], func=func, b=b, B=B)

    Note that we set :math:`m = 10` since we have

    * :math:`m_p=2` polynomial basis functions :math:`\\boldsymbol{\\phi}(x) =
      (1, x)` of order 0 and 1.
    * :math:`2m_t` trigonometric basis functions with :math:`m_t=2` as
      :math:`\\boldsymbol{\\phi} = (\\sin(x), \\sin(2x), \\cos(x), \\cos(2x))`.
    * :math:`2m_h` hyperbolic basis functions with :math:`m_h=1` as
      :math:`\\boldsymbol{\\phi}=(\\sinh(3x),\\cosh(3x))`.
    * :math:`m_f=2` user-defined basis functions :math:`\\boldsymbol{\\phi}(x)
      = (\\frac{1}{2}(3x^3-1), \\frac{1}{2}(5x^3-3x))`.

    Recall, the total number of basis functions is
    :math:`m = m_p + 2m_t + 2m_h + m_f`, so in total there are 10 basis
    functions. Because of this, the vector ``b`` and matrix ``B`` were defined
    with the size :math:`m=10`.

    **Generate Design Matrix:**

    The design matrix :math:`\\mathbf{X}` can be accessed by ``X`` attribute:

    .. code-block:: python

        >>> # Get design matrix on data points
        >>> X = f.X
        >>> X.shape
        (30, 10)

    Alternatively, the design matrix :math:`\\mathbf{X}^{\\ast}` can be
    generated on arbitrary test points :math:`\\boldsymbol{x}^{\\ast}` as
    follows:

    .. code-block:: python

        >>> # Generate 100 test points
        >>> x_test = generate_points(num_points=100, dimension=1)

        >>> # Generate design matrix on test points
        >>> X_test = f.generate_design_matrix(x_test)
        >>> X_test.shape
        (100, 10)

    **Orthonormalize Basis Functions:**

    Repeat the above example but set ``orthonormalize`` to `True`:

    .. code-block:: python
        :emphasize-lines: 5

        >>> # Create basis functions
        >>> f = LinearModel(x, polynomial_degree=1,
        ...                 trigonometric_coeff=[1.0, 2.0],
        ...                 hyperbolic_coeff=[3.0], func=func,
        ...                 orthonormalize=True)

        >>> # Check the orthonormality of X
        >>> I = numpy.eye(m)
        >>> numpy.allclose(f.X.T @ f.X, I, atol=1e-6)
        True

        >>> # Generate design matrix on test points
        >>> X_test = f.generate_design_matrix(x_test, orthonormalize=True)

        >>> # Check the orthonormality of X
        >>> numpy.allclose(X_test.T @ X_test, I, atol=1e-6)
        True
    """

    # ====
    # init
    # ====

    def __init__(
            self,
            x,
            polynomial_degree=0,
            trigonometric_coeff=None,
            hyperbolic_coeff=None,
            func=None,
            orthonormalize=False,
            b=None,
            B=None):
        """
        """

        trigonometric_coeff, hyperbolic_coeff = self._check_arguments(
                x, polynomial_degree, trigonometric_coeff, hyperbolic_coeff,
                func)

        # If points are 1d array, wrap them to a 2d array
        if x.ndim == 1:
            x = numpy.array([x], dtype=float).T

        # Store function info
        self.points = x
        self.polynomial_degree = polynomial_degree
        self.trigonometric_coeff = trigonometric_coeff
        self.hyperbolic_coeff = hyperbolic_coeff
        self.func = func
        self.orthonormalize = orthonormalize

        # Generate design matrix
        self.X = self.generate_design_matrix(self.points, self.orthonormalize)

        # Check b and B for their size consistency with X
        b, B = self._check_b_B(b, B)

        self.b = b        # Prior mean of beta
        self.B = B        # Prior covariance of beta
        self.beta = None  # Posterior mean of beta (will be computed)
        self.C = None     # Posterior covariance of beta (will be computed)

        # Precision of the prior of beta
        if self.B is not None:
            self.Binv = numpy.linalg.inv(self.B)
        else:
            # When B is None, we assume it is infinity. Hence, the precision
            # matrix (inverse of covariance) will be zero matrix.
            m = self.X.shape[1]
            self.Binv = numpy.zeros((m, m), dtype=float)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(
            self,
            x,
            polynomial_degree,
            trigonometric_coeff,
            hyperbolic_coeff,
            func):
        """
        """

        # Check x
        if x is None:
            raise ValueError('"x" cannot be "None".')

        elif not isinstance(x, numpy.ndarray):
            raise TypeError('"x" should be a "numpy.ndarray" type.')

        # Check at least one of polynomial, trigonometric, hyperbolic, or func
        # is given.
        if (polynomial_degree is None) and (trigonometric_coeff is None) and \
           (hyperbolic_coeff is None) and (func is None):
            raise ValueError('At least, one of "polynomial_degree", ' +
                             '"trigonometric_coeff", "hyperbolic_coeff", ' +
                             'or "func" must be set.')

        # Check polynomial degree
        if polynomial_degree is not None:

            if not isinstance(polynomial_degree, int):
                raise ValueError('"polynomial_degree" must be an integer.')

            elif polynomial_degree < 0:
                raise ValueError('"polynomial_degree" should be non-negative.')

        # Check trigonometric coeff
        if trigonometric_coeff is not None:

            if numpy.isscalar(trigonometric_coeff):
                if not isinstance(trigonometric_coeff, int) and \
                   not isinstance(trigonometric_coeff, float):
                    raise ValueError('"trigonometric_coeff" must be a float ' +
                                     'type.')

                # Convert scalar to numpy array
                trigonometric_coeff = numpy.array([trigonometric_coeff],
                                                  dtype=float)

            elif isinstance(trigonometric_coeff, list):
                # Convert list to numpy array
                trigonometric_coeff = numpy.array(trigonometric_coeff,
                                                  dtype=float)
            elif not isinstance(trigonometric_coeff, numpy.ndarray):
                raise TypeError('"trigonometric_coeff" should be a scalar, ' +
                                ', a list, or an array.')
            elif trigonometric_coeff.ndim > 1:
                raise ValueError('"trigonometric_coeff" should be a 1d array.')

        # Check polynomial degree
        if hyperbolic_coeff is not None:

            if numpy.isscalar(hyperbolic_coeff):
                if not isinstance(hyperbolic_coeff, int) and \
                   not isinstance(hyperbolic_coeff, float):
                    raise ValueError('"hyperbolic_coeff" must be a float ' +
                                     'type.')

                # Convert scalar to numpy array
                hyperbolic_coeff = numpy.array([hyperbolic_coeff], dtype=float)

            elif isinstance(hyperbolic_coeff, list):
                # Convert list to numpy array
                hyperbolic_coeff = numpy.array(hyperbolic_coeff, dtype=float)
            elif not isinstance(hyperbolic_coeff, numpy.ndarray):
                raise TypeError('"hyperbolic_coeff" should be a scalar, ' +
                                ', a list, or an array.')
            elif hyperbolic_coeff.ndim > 1:
                raise ValueError('"hyperbolic_coeff" should be a 1d array.')

        # Check func
        if func is not None and not callable(func):
            raise TypeError('"func" should be a callable function or None.')

        return trigonometric_coeff, hyperbolic_coeff

    # =========
    # check b B
    # =========

    def _check_b_B(self, b, B):
        """
        Checks the sizes of and B to be consistent with self.X.
        """

        # Check b
        if b is not None:
            if numpy.isscalar(b):

                if self.X.ndim != 1 or self.X.shape[1] != 1:
                    raise ValueError('"b" should be a vector of the same ' +
                                     'size as the number of columns of the ' +
                                     'design matrix "X".')
                else:
                    # Convert scalar to a 1d vector of unit size
                    b = numpy.array([b], dtype=float)

            elif b.size != self.X.shape[1]:
                raise ValueError('"b" should have the same size as the ' +
                                 'number of columns of the design matrix ' +
                                 '"X", which is %d.' % self.X.shape[1])

            if not isinstance(b, numpy.ndarray):
                raise TypeError('"b" should be either a scalar (if the' +
                                'design matrix is a column vector) or a row' +
                                'vector of "numpy.ndarray" type.')

        # Check B
        if B is not None:

            if b is None:
                raise ValueError('When "B" is given, "b" cannot be None.')
            elif numpy.isscalar(b) and not numpy.isscalar(B):
                raise ValueError('When "b" is a scalar, "B" should also be a' +
                                 'scalar.')
            elif not isinstance(B, numpy.ndarray):
                raise TypeError('"B" should be a "numpy.ndarray" type.')

            elif numpy.isscalar(B):
                if self.X.ndim != 1 or self.X.shape[1] != 1:
                    raise ValueError('"b" should be a vector of the same ' +
                                     'size as the number of columns of the ' +
                                     'design matrix "X".')
                else:
                    # Convert scalar to a 2d vector of unit size
                    B = numpy.array([[B]], dtype=float)

            elif B.shape != (b.size, b.size):
                raise ValueError('"B" should be a square matrix with the' +
                                 'same number of columns/rows as the size of' +
                                 'vector "b".')

        return b, B

    # ======================
    # generate design matrix
    # ======================

    def generate_design_matrix(self, x, orthonormalize=False):
        """
        Generates design matrix on test points.

        .. note::

            The design matrix on *data points* is automatically generated
            during the internal training process and can be accessed by ``X``
            attribute. Use this function to generate the design matrix on
            arbitrary *test points*.

        Parameters
        ----------

        x : numpy.ndarray
            A 2D array of data points where each row of the array is the
            coordinate of a point :math:`\\boldsymbol{x}=(x_1, \\dots, x_d)`.
            The array size is :math:`n \\times d` where :math:`n` is the number
            of the points and :math:`d` is the dimension of the space of
            points.

        orthonormalize : bool, default=False
            If `True`, the design matrix :math:`\\mathbf{X}` of the basis
            functions will be orthonormalized.

        Returns
        -------

        X : numpy.ndarray[float]
            Design matrix of the size :math:`n \\times m` where :math:`n` is
            the number of points and :math:`m` is the number of basis
            functions.

        Notes
        -----

        This function generates the design matrix :math:`\\mathbf{X}^{\\ast}`
        on a set of test points
        :math:`\\{ \\boldsymbol{x}^{\\ast}_i \\}_{i=1}^n` with the components
        :math:`X_{ij}^{\\ast}` defined as

        .. math::

            X_{ij}^{\\ast} = \\phi_j(\\boldsymbol{x}_i).

        The size of the matrix is :math:`n \\times m` where :math:`n` is the
        number of test points and :math:`m` is the total number of basis
        functions, which can be obtained by

        .. math::

            m = m_p + 2m_t + 2m_h + m_f,

        where

        * :math:`m_p` is the number of polynomial basis functions.
        * :math:`m_t` is the number of coefficients of the trigonometric
          functions.
        * :math:`m_h` is the number of coefficients of the hyperbolic
          functions.
        * :math:`m_h` is the number of user-defined basis functions.

        **Orthonormalization:**

        If ``orthonormalize`` is set to `True`, the output matrix
        :math:`\\mathbf{X}^{\\ast}` is orthonormalized so that

        .. math::

            (\\mathbf{X}^{\\ast})^{\\intercal} \\mathbf{X}^{\\ast} =
            \\mathbf{I},

        where :math:`\\mathbf{I}` is the :math:`m \\times m` identity matrix.

        This function uses :func:`detkit.orthogonalize` function to
        orthonormalize the matrix :math:`\\mathbf{X}`.

        Examples
        --------

        First, create a set of 50 random points in the interval :math:`[0, 1]`.

        .. code-block:: python

            >>> # Generate a set of points
            >>> from glearn.sample_data import generate_points
            >>> x = generate_points(num_points=30, dimension=1)

        Define the function :math:`\\boldsymbol{\\phi}` as follows:

        .. code-block:: python

            >>> # Create a function returning the Legendre Polynomials
            >>> import numpy
            >>> def func(x):
            ...     phi_1 = 0.5 * (3.0*x**2 - 1)
            ...     phi_2 = 0.5 * (5.0*x**3 - 3.0*x)
            ...     phi = numpy.array([phi_1, phi_2])
            ...     return phi

        **Create Linear Model:**

        Create a linear model with first order polynomial basis functions, both
        trigonometric and hyperbolic functions, and the user-defined functions
        created in the above.

        .. code-block:: python

            >>> # Create basis functions
            >>> from glearn import LinearModel
            >>> f = LinearModel(x, polynomial_degree=1,
            ...                 trigonometric_coeff=[1.0, 2.0],
            ...                 hyperbolic_coeff=[3.0], func=func)

        **Generate Design Matrix:**

        Generate the design matrix :math:`\\mathbf{X}^{\\ast}` on arbitrary
        test points :math:`\\boldsymbol{x}^{\\ast}`:

        .. code-block:: python

            >>> # Generate 100 test points
            >>> x_test = generate_points(num_points=100, dimension=1)

            >>> # Generate design matrix on test points
            >>> X_test = f.generate_design_matrix(x_test)
            >>> X_test.shape
            (100, 10)

        **Orthonormalize Basis Functions:**

        Repeat the above example but set ``orthonormalize`` to `True`:

        .. code-block:: python
            :emphasize-lines: 2

            >>> # Generate design matrix on data points
            >>> X = f.generate_design_matrix(x, orthonormalize=True)

            >>> # Check the orthonormality of X
            >>> I = numpy.eye(X_test.shape[1])
            >>> numpy.allclose(X_test.T @ X_test, I, atol=1e-6)
            True
        """

        # Convert a vector to matrix if dimension is one
        if x.ndim == 1:
            x = numpy.array([x]).T

        # Initialize output
        X_list = []

        # Polynomial basis functions
        if self.polynomial_degree is not None:
            X_polynomial = self._generate_polynomial_basis(x)
            X_list.append(X_polynomial)

        # Trigonometric basis functions
        if self.trigonometric_coeff is not None:
            X_trigonometric = self._generate_trigonometric_basis(x)
            X_list.append(X_trigonometric)

        # Hyperbolic basis functions
        if self.hyperbolic_coeff is not None:
            X_hyperbolic = self._generate_hyperbolic_basis(x)
            X_list.append(X_hyperbolic)

        # Custom function basis functions
        if self.func is not None:
            X_fun = self._generate_custom_basis(x)
            X_list.append(X_fun)

        # Concatenate those bases that are not None
        if len(X_list) == 0:
            raise RuntimeError('No basis was generated.')
        elif len(X_list) == 1:
            X = X_list[0]
        else:
            X = numpy.concatenate(X_list, axis=1)

        # Orthonormalize
        if orthonormalize:
            orthogonalize(X)

        return X

    # =========================
    # generate polynomial basis
    # =========================

    def _generate_polynomial_basis(self, x):
        """
        Generates polynomial basis functions.
        """

        n = x.shape[0]
        dimension = x.shape[1]

        # Adding polynomial functions
        powers_array = numpy.arange(self.polynomial_degree + 1)
        powers_tile = numpy.tile(powers_array, (dimension, 1))
        powers_mesh = numpy.meshgrid(*powers_tile)

        powers_ravel = []
        for i in range(dimension):
            powers_ravel.append(powers_mesh[i].ravel())

        # The array powers_ravel contains all combinations of powers
        powers_ravel = numpy.array(powers_ravel)

        # For each combination of powers, we compute the power sum
        powers_sum = numpy.sum(powers_ravel, axis=0)

        # The array powers contains those combinations that their sum does
        # not exceed the polynomial_degree
        powers = powers_ravel[:, powers_sum <= self.polynomial_degree]

        num_degrees = powers.shape[0]
        num_basis = powers.shape[1]

        # Basis functions
        X_polynomial = numpy.ones((n, num_basis), dtype=float)
        for j in range(num_basis):
            for i in range(num_degrees):
                X_polynomial[:, j] *= x[:, i]**powers[i, j]

        return X_polynomial

    # ============================
    # generate trigonometric basis
    # ============================

    def _generate_trigonometric_basis(self, x):
        """
        Generates trigonometric basis functions.
        """

        n = x.shape[0]
        dimension = x.shape[1]

        tri_size = self.trigonometric_coeff.size
        X_trigonometric = numpy.empty((n, 2*dimension*tri_size))

        for i in range(tri_size):
            for j in range(dimension):
                X_trigonometric[:, 2*dimension*i + 2*j+0] = numpy.sin(
                        x[:, j] * self.trigonometric_coeff[i])
                X_trigonometric[:, 2*dimension*i + 2*j+1] = numpy.cos(
                        x[:, j] * self.trigonometric_coeff[i])

        return X_trigonometric

    # =========================
    # generate hyperbolic basis
    # =========================

    def _generate_hyperbolic_basis(self, x):
        """
        Generate hyperbolic basis functions.
        """

        n = x.shape[0]
        dimension = x.shape[1]

        hyp_size = self.hyperbolic_coeff.size
        X_hyperbolic = numpy.empty((n, 2*dimension*hyp_size))

        for i in range(hyp_size):
            for j in range(dimension):
                X_hyperbolic[:, 2*dimension*i + 2*j+0] = numpy.sinh(
                        x[:, j] * self.hyperbolic_coeff[i])
                X_hyperbolic[:, 2*dimension*i + 2*j+1] = numpy.cosh(
                        x[:, j] * self.hyperbolic_coeff[i])

        return X_hyperbolic

    # =====================
    # generate custom basis
    # =====================

    def _generate_custom_basis(self, x):
        """
        Generate custom basis functions.
        """

        n = x.shape[0]

        # Generate on the first point to see what is the size of the output
        X_fun_init = numpy.squeeze(self.func(x[0, :]))

        if X_fun_init.ndim != 1:
            raise ValueError('"func" should output 1d array.')

        # Initialize output 2D array for all points
        X_fun = numpy.empty((n, X_fun_init.size), dtype=float)
        X_fun[0, :] = X_fun_init

        for i in range(1, n):
            X_fun[i, :] = numpy.squeeze(self.func(x[i, :]))

        return X_fun

    # =================
    # update hyperparam
    # =================

    def update_hyperparam(self, cov, y):
        """
        Manually update the posterior mean and covariance of linear model
        coefficient.

        .. note::

            This function is automatically called once the Gaussian process is
            trained after calling :meth:`glearn.GaussianProcess.train`. Hence,
            there is no need to call this function unless the user wants to
            manually update the hyperparameters.

        Parameters
        ----------

        cov : glearn.Covariance
            Covariance object of the Gaussian process regression.

        y : numpy.array
            Array of training data.

        See Also
        --------

        :meth:`glearn.GaussianProcess.train`

        Notes
        -----

        **Before Training:**

        Before training the Gaussian process, the coefficient of the linear
        model, :math:`\\boldsymbol{\\beta}`, is unknown, however, it is
        specified by a prior distribution

        .. math::

            \\boldsymbol{\\beta} \\sim \\mathcal{N}(\\boldsymbol{b},
            \\sigma^2 \\mathbf{B}),

        where :math:`\\boldsymbol{b}` and :math:`\\mathbf{B}` are given by the
        user and :math:`\\sigma` is unknown.

        **After Training:**

        Once the function :meth:`glearn.GaussianProcess.train` is called to
        train the model, a posterior distribution for the coefficient
        :math:`\\boldsymbol{\\beta}` is readily available as

        .. math::

            \\boldsymbol{\\beta} \\sim \\mathcal{N}(
            \\hat{\\boldsymbol{\\beta}}, \\mathbf{C}),

        where

        * :math:`\\hat{\\boldsymbol{\\beta}}` is the posterior mean and can be
          accessed by ``LinearModel.beta`` attribute.
        * :math:`\\mathbf{C}` is the posterior covariance and can be accessed
          by ``LinearModel.C`` attribute.

        The user can, however, manually update the above posterior parameters
        by calling this function.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 25

            >>> # Import modules
            >>> import glearn
            >>> from glearn import sample_data

            >>> # Generate sample points
            >>> x = sample_data.generate_points(num_points=50)

            >>> # Generate noisy data
            >>> y = sample_data.generate_data(x, noise_magnitude=0.05)

            >>> # Create a linear model
            >>> mean = glearn.LinearModel(x)

            >>> # Create a covariance model
            >>> cov = glearn.Covariance(x)

            >>> # Create a Gaussian process model
            >>> gp = glearn.GaussianProcess(mean, cov)

            >>> # Train model with data
            >>> gp.train(y)

            >>> # Update hyperparameter of the linear model
            >>> # Note that this is done automatically.
            >>> mean.update_hyperparam(cov, y)

            >>> # Get the updated posterior mean of beta
            >>> mean.beta
            [0.0832212]

            >>> # Get the updated posterior covariance of beta
            >>> mean.C
            [[0.79492599]]
        """

        # Note: cov should has been updated already after training.
        sigma, sigma0 = cov.get_sigmas()

        # Posterior covariance of beta
        Y = cov.solve(self.X, sigma=sigma, sigma0=sigma0)
        Cinv = numpy.matmul(self.X.T, Y)

        # Note: B in this class is B1 in the paper notations. That is, self.B
        # here means B1 = B / (sigma**2).
        if self.B is not None:
            Cinv += self.Binv / (sigma**2)

        self.C = numpy.linalg.inv(Cinv)

        # Posterior mean of beta
        v = numpy.dot(Y.T, y)
        if self.B is not None:
            Binvb = numpy.dot(self.Binv, self.b) / (sigma**2)
            v += Binvb
        self.beta = numpy.matmul(self.C, v)
