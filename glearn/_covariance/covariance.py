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
    Mixed covariance model.

    For the regression problem
    :math:`y=\\mu(\\boldsymbol{x})+\\delta(\\boldsymbol{x}) + \\epsilon` where
    :math:`\\mu` and :math:`\\delta` are the mean, miss-fit, and input noise of
    of the regression model respectively, this class implements a covariance
    model

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
        array of the size :math:`d` is given, :math:`\\alpha_i` scales the
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

    * :math:``\\mu` is a deterministic mean function.
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
            if not isinstance(sigma, int) and isinstance(sigma, float):
                raise TypeError('"sigma" should be a float type.')
            elif sigma < 0.0:
                raise ValueError('"sigma" cannot be negative.')

        # Check sigma0
        if sigma0 is not None:
            if not isinstance(sigma0, int) and isinstance(sigma0, float):
                raise TypeError('"sigma0" should be a float type.')
            elif sigma0 < 0.0:
                raise ValueError('"sigma0" cannot be negative.')

    # =================
    # set imate options
    # =================

    def set_imate_options(self, imate_options):
        """
        Updates the ``imate_options`` attribute.

        .. note::

            This function is intended to be used internally.

        Parameters
        ----------

        imate_options : dict
            A dictionary of options to be passed to the functions in
            `imate <https://ameli.github.io/imate/index.html>`_ package.

        Notes
        -----

        This function updates the attribute ``imate_options`` for the
        instance of the class :class:`glearn._covariance.MixedCorrelation`
        object. The existing options in the dictionary ``imate_option`` are
        overwritten, and new options will be added (if they do not already
        exist in the current dictionary).
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
        Sets the scale attribute of correlation matrix.
        """

        self.mixed_cor.set_scale(scale)

    # =========
    # get scale
    # =========

    def get_scale(self):
        """
        Returns distance scale of self.mixed_cor.cor object.
        """

        return self.mixed_cor.get_scale()

    # ==========
    # set sigmas
    # ==========

    def set_sigmas(self, sigma, sigma0):
        """
        After training, when optimal sigma and sigma0 is obtained, this
        function stores sigma and sigma0 as attributes of the class.
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
        Returns sigma and sigma0. If the inputs are None, the object attributes
        are used.

        After training, when optimal sigma and sigma0 are obtained and set as
        the attributes of this class, the next calls to other functions like
        solve, trace, traceinv, etc, should use the optimal sigma and sigma0.
        Thus, we will call these functions without specifying sigma, and sigma0
        and this function returns the sigma and sigma0 that are stored as
        attributes.
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
        Get the matrix as a numpy array of scipy sparse array.
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
            exponent=1,
            derivative=[],
            imate_options={}):
        """
        Computes

        .. math::

            \\mathrm{trace} \\frac{\\partial^q}{\\partial \\theta^q}
            (\\sigma^2 \\mathbf{K} + \\sigma_0^2 \\mathbf{I})^{p},

        where

        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`p`is a non-negative integer.
        * :math:`\\sigma` and :math:`\\sigma_0` are real numbers.
        * :math:`\\theta` is correlation scale parameter.
        * :math:`q` is the order of the derivative.
        """

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if (exponent > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "exponent" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and exponent == 0:
            # Matrix is zero.
            trace_ = 0.0

        elif exponent == 0:
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
                trace_ = (sigma0**(2.0*exponent)) * n

        else:
            # Derivative eliminates sigma0^2 I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            trace_ = sigma**(2.0*exponent) * self.mixed_cor.trace(
                    eta, scale, exponent, derivative, imate_options)

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
            exponent=1,
            derivative=[],
            imate_options={}):
        """
        Computes

        .. math::

            \\mathrm{trace} \\left( \\frac{\\partial^q}{\\partial \\theta^q}
            (\\sigma^2 \\mathbf{K} + \\sigma_0^2 \\mathbf{I})^{-p} \\mathbf{B}
            \\right)

        where

        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`p`is a non-negative integer.
        * :math:`\\sigma` and :math:`\\sigma_0` are real numbers.
        * :math:`\\theta` is correlation scale parameter.
        * :math:`q` is the order of the derivative.
        * :math:`\\mathbf{B}` is a matrix. If set to None, identity matrix is
          assumed.
        """

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if (B is None) and (C is not None):
            raise ValueError('When "C" is given, "B" should also be given.')

        if (exponent > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "exponent" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and exponent == 0:
            # Matrix is zero.
            traceinv_ = numpy.nan

        elif exponent == 0:
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
                    traceinv_ = n / (sigma0**(2.0*exponent))
                else:
                    # B is not identity
                    if C is None:
                        traceinv_ = imate.trace(B, method='exact') / \
                                (sigma0**(2.0*exponent))
                    else:
                        # C is not indentity. Compute trace of C*B devided by
                        # sigma0**4 (becase when we have C, there are to
                        # matrix A).
                        if isspmatrix(C):
                            traceinv_ = numpy.sum(C.multiply(B.T).data) / \
                                    (sigma0**(4.0*exponent))
                        else:
                            traceinv_ = numpy.sum(numpy.multiply(C, B.T)) / \
                                    (sigma0**(4.0*exponent))

        else:
            # Derivative eliminates sigma0^2*I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            traceinv_ = self.mixed_cor.traceinv(
                    eta, B, C, scale, exponent, derivative, imate_options)
            if C is None:
                traceinv_ /= sigma**(2.0*exponent)
            else:
                # When C is given, there are two A matrices (C*Ainv*B*Ainv)
                traceinv_ /= sigma**(4.0*exponent)

        return traceinv_

    # ======
    # logdet
    # ======

    def logdet(
            self,
            sigma=None,
            sigma0=None,
            scale=None,
            exponent=1,
            derivative=[],
            imate_options={}):
        """
        Computes

        .. math::

            \\mathrm{det} \\frac{\\partial^q}{\\partial \\theta^q}
            (\\sigma^2 \\mathbf{K} + \\sigma_0^2 \\mathbf{I})^{p},

        where

        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`p`is a non-negative integer.
        * :math:`\\sigma` and :math:`\\sigma_0` are real numbers.
        * :math:`\\theta` is correlation scale parameter.
        * :math:`q` is the order of the derivative.
        """

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if (exponent > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "exponent" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and exponent == 0:
            # Matrix is zero.
            logdet_ = -numpy.inf

        elif exponent == 0:
            # Matrix is identity.
            logdet_ = 0.0

        elif numpy.abs(sigma) < self.tol:

            n = self.mixed_cor.get_matrix_size()

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                logdet_ = -numpy.inf
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                logdet_ = (2.0*exponent*n) * numpy.log(sigma0)

        else:
            n = self.mixed_cor.get_matrix_size()

            # Derivative eliminates sigma0^2 I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            logdet_ = (2.0*exponent*n) * numpy.log(sigma) + \
                self.mixed_cor.logdet(eta, scale, exponent, derivative,
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
            exponent=1,
            derivative=[]):
        """
        Solves the linear system

        .. math::

            \\frac{\\partial^q}{\\partial \\theta^q}
            (\\sigma^2 \\mathbf{K} + \\sigma_0^2 \\mathbf{I})^{p} \\mathbf{X}
            = \\mathbf{Y},

        where:

        * :math:`\\mathbf{Y}` is the given right hand side matrix,
        * :math:`\\mathbf{X}` is the solution (unknown) matrix,
        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`p`is a non-negative integer.
        * :math:`\\sigma` and :math:`\\sigma_0` are real numbers.
        * :math:`\\theta` is correlation scale parameter.
        * :math:`q` is the order of the derivative.
        """

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if (exponent > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "exponent" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif len(derivative) > 0 and exponent == 0:
            # Matrix is zero, hence has no inverse.
            X = numpy.zeros_like(Y)
            X[:] = numpy.nan

        elif exponent == 0:
            # Matrix is identity.
            X = numpy.copy(Y)

        elif numpy.abs(sigma) < self.tol:

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                X = numpy.zeros_like(Y)
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                X = Y / (sigma0**(2*exponent))

        else:
            # Derivative eliminates sigma0^2 I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            X = self.mixed_cor.solve(
                    Y, eta, scale, exponent, derivative) / \
                (sigma**(2*exponent))

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
            exponent=1,
            derivative=[]):
        """
        Matrix-vector multiplication:

        .. math::

            \\boldsymbol{y} = \\frac{\\partial^q}{\\partial \\theta^q}
            (\\sigma^2 \\mathbf{K}(\\theta) + \\sigma_0^2 \\mathbf{I})^{p}
            \\boldsymbol{x}

        where:

        * :math:`\\boldsymbol{x}` is the given vector,
        * :math:`\\boldsymbol{y}` is the product vector,
        * :math:`\\mathbf{I}` is the identity matrix,
        * :math:`\\sigma` and :math:`\\sigma_0` are real numbers.
        * :math:`p`is a non-negative integer.
        * :math:`\\theta` is correlation scale parameter.
        * :math:`q` is the order of the derivative.
        """

        # Get sigma and sigma0 (if None, uses class attribute)
        sigma, sigma0 = self.get_sigmas(sigma, sigma0)

        if (exponent > 1) and (len(derivative) > 0):
            raise NotImplementedError('If "exponent" is larger than one, ' +
                                      '"derivative" should be zero (using ' +
                                      'an empty list).')

        elif exponent == 0 and len(derivative) > 0:
            # Matrix is zero.
            y = numpy.zeros_like(x)

        elif exponent == 0:
            # Matrix is identity.
            y = x.copy()

        elif numpy.abs(sigma) < self.tol:

            if len(derivative) > 0:
                # mixed covariance is independent of derivative parameter
                y = numpy.zeros_like(x)
            else:
                # Ignore (sigma**2 * K) compared to (sigma0**2 * I) term.
                y = sigma0**(2.0*exponent) * x

        else:
            # Derivative eliminates sigma0^2 I term.
            if len(derivative) > 0:
                sigma0 = 0.0

            eta = (sigma0 / sigma)**2
            y = (sigma**(2.0*exponent)) * \
                self.mixed_cor.dot(x, eta, scale, exponent, derivative)

        return y

    # ===============
    # auto covariance
    # ===============

    def auto_covariance(self, test_points):
        """
        Computes the auto-covariance between the training points and
        themselves.
        """

        if self.sigma is None:
            raise RuntimeError('"sigma" cannot be None to create auto ' +
                               'covariance.')

        auto_cor = self.cor.auto_correlation(test_points)
        auto_cov = (self.sigma**2) * auto_cor

        return auto_cov

    # ================
    # cross covariance
    # ================

    def cross_covariance(self, test_points):
        """
        Computes the cross-covariance between the training points (points
        which this object is initialized with), and a given set of test points.
        This matrix is rectangular.
        """

        if self.sigma is None:
            raise RuntimeError('"sigma" cannot be None to create cross ' +
                               'covariance.')

        cross_cor = self.cor.cross_correlation(test_points)
        cross_cov = (self.sigma**2) * cross_cor

        return cross_cov
