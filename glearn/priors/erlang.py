# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import numpy
from scipy.special import gamma
from .prior import Prior

__all__ = ['Erlang']


# ======
# Erlang
# ======

class Erlang(Prior):
    """
    Erlang distribution.

    .. note::

        For the methods of this class, see the base class
        :class:`glearn.priors.Prior`.

    Parameters
    ----------

    shape : float or array_like[float], default=1
        The shape parameter :math:`\\alpha` of Erlang distribution. If an array
        :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_p)` is
        given, the prior is assumed to be :math:`p` independent Erlang
        distributions each with shape :math:`\\alpha_i`.

    rate : float or array_like[float], default=1.0
        The rate :math:`\\beta` of Erlang distribution. If an array
        :math:`\\boldsymbol{\\beta} = (\\beta_1, \\dots, \\beta_p)` is given,
        the prior is assumed to be :math:`p` independent Erlang distributions
        each with rate :math:`\\beta_i`.

    Attributes
    ----------

    shape : float or array_like[float], default=0
        Shape parameter :math:`\\alpha` of the distribution

    rate : float or array_like[float], default=0
        Rate parameter :math:`\\beta` of the distribution

    Methods
    -------

    suggest_hyperparam
    pdf
    pdf_jacobian
    pdf_hessian

    See Also
    --------

    glearn.priors.Prior

    Notes
    -----

    **Single Hyperparameter:**

    The Erlang distribution with shape parameter :math:`\\alpha > 0` and rate
    parameter :math:`\\beta > 0` is defined by the probability density function

    .. math::

        p(\\theta \\vert \\alpha, \\beta) = \\frac{\\theta^{\\alpha-1}
        e^{-\\beta \\theta} \\beta^{\\alpha}}{(\\alpha -1)!}.

    .. note::

        The Erlang distribution is the same as Gamma distribution when
        :math:`\\alpha` is an integer. For non-integer :math:`\\alpha`, use
        the Gamma distribution :class:`glearn.priors.Gamma`.

    **Multiple Hyperparameters:**

    If an array of the hyperparameters are given, namely
    :math:`\\boldsymbol{\\theta} = (\\theta_1, \\dots, \\theta_p)`, then
    the prior is the product of independent priors

    .. math::

        p(\\boldsymbol{\\theta}) = p(\\theta_1) \\dots p(\\theta_p).

    In this case, if the input arguments ``shape`` and ``rate`` are given as
    the arrays :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_p)`
    and :math:`\\boldsymbol{\\beta} = (\\beta_1, \\dots, \\beta_p)`, each prior
    :math:`p(\\theta_i)` is defined as the Erlang distribution with shape
    parameter :math:`\\alpha_i` and rate parameter :math:`\\beta_i`. In
    contrary, if ``shape`` and ``rate`` are given as the scalars
    :math:`\\alpha` and :math:`\\beta`, then all priors :math:`p(\\theta_i)`
    are defined as the Erlang distribution with shape parameter :math:`\\alpha`
    and rate parameter :math:`\\beta`.

    Examples
    --------

    **Create Prior Objects:**

    Create the Erlang distribution with the shape parameter :math:`\\alpha=2`
    and rate parameter :math:`\\beta=4`.

    .. code-block:: python

        >>> from glearn import priors
        >>> prior = priors.Erlang(2, 4)

        >>> # Evaluate PDF function at multiple locations
        >>> t = [0, 0.5, 1]
        >>> prior.pdf(t)
        array([0.        , 1.08268227, 0.29305022])

        >>> # Evaluate the Jacobian of the PDF
        >>> prior.pdf_jacobian(t)
        array([        nan, -2.16536453, -0.87915067])

        >>> # Evaluate the Hessian of the PDF
        >>> prior.pdf_hessian(t)
        array([[       nan, 0.        , 0.        ],
               [0.        , 0.        , 0.        ],
               [0.        , 0.        , 2.34440178]])

        >>> # Evaluate the log-PDF
        >>> prior.log_pdf(t)
        -44.87746683446311

        >>> # Evaluate the Jacobian of the log-PDF
        >>> prior.log_pdf_jacobian(t)
        array([ -6.90775528, -26.82306851, -89.80081863])

        >>> # Evaluate the Hessian of the log-PDF
        >>> prior.log_pdf_hessian(t)
        array([[ -21.20759244,    0.        ,    0.        ],
               [   0.        ,  -67.06429581,    0.        ],
               [   0.        ,    0.        , -212.07592442]])

        >>> # Plot the distribution and its first and second derivative
        >>> prior.plot()

    .. image:: ../_static/images/plots/prior_erlang.png
        :align: center
        :width: 100%
        :class: custom-dark

    **Where to Use the Prior object:**

    Define a covariance model (see :class:`glearn.Covariance`) where its scale
    parameter is a prior function.

    .. code-block:: python
        :emphasize-lines: 7

        >>> # Generate a set of sample points
        >>> from glearn.sample_data import generate_points
        >>> points = generate_points(num_points=50)

        >>> # Create covariance object of the points with the above kernel
        >>> from glearn import covariance
        >>> cov = glearn.Covariance(points, kernel=kernel, scale=prior)
    """

    # ====
    # init
    # ====

    def __init__(self, shape=1, rate=1.0):
        """
        Initialization.
        """

        # This distribution does not have half distribution
        half = False
        super().__init__(half)

        # Check arguments
        self.shape, self.rate = self._check_arguments(shape, rate)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, shape, rate):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check shape
        if numpy.isscalar(shape) and \
                not isinstance(shape, (int, numpy.integer)):
            raise ValueError('"shape" should be an integer. For non-integer ' +
                             '"shape" parameter, use "Gamma" distribution.')
        if any(numpy.array([shape]) <= 0.0):
            raise ValueError('"shape" should be positive.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(shape):
            shape = numpy.array([shape], dtype=int)
        elif isinstance(shape, list):
            shape = numpy.array(shape, dtype=int)
        elif not isinstance(shape, numpy.ndarray):
            raise TypeError('"shape" should be a scalar, list of numpy ' +
                            'array.')

        # Check rate
        if numpy.isscalar(rate) and not isinstance(rate, (int, float)):
            raise TypeError('"rate" should be a float number.')
        if any(numpy.array([rate]) <= 0.0):
            raise ValueError('"rate" should be positive.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(rate):
            rate = numpy.array([rate], dtype=float)
        elif isinstance(rate, list):
            rate = numpy.array(rate, dtype=float)
        elif not isinstance(rate, numpy.ndarray):
            raise TypeError('"rate" should be a scalar, list of numpy array.')

        return shape, rate

    # ==================
    # suggest hyperparam
    # ==================

    def suggest_hyperparam(self, positive=False):
        """
        Find an initial guess for the hyperparameters based on the peaks of the
        prior distribution.

        Parameters
        ----------

        positive : bool, default=False
            If `True`, it suggests a positive hyperparameter. This is used
            for instance if the suggested hyperparameter is used for the
            scale parameter which should always be positive.

            .. note::
                This parameter is not used, rather, ignored in this function.
                This parameter is included for consistency this function with
                the other prior classes.

        Returns
        -------

        hyperparam : float or numpy.array[float]
            A feasible guess for the hyperparameter. The output is either a
            scalar or an array of the same size as the input parameters of the
            distribution.

        See Also
        --------

        glearn.GaussianProcess

        Notes
        -----

        For the Erlang distribution with shape parameter :math:`\\alpha` and
        rate parameter :math:`\\beta`, the suggested hyperparameter is the mean
        :math:`\\mu` of the distribution defined by

        .. math::

            \\mu = \\frac{\\alpha}{\\beta}.

        The suggested hyperparameters can be used as initial guess for the
        optimization of the posterior functions when used with this prior.

        Examples
        --------

        Create the Erlang distribution with the shape parameter
        :math:`\\alpha=2` and rate parameter :math:`\\beta=4`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Erlang(2, 4)

            >>> # Find a feasible hyperparameter value
            >>> prior.suggest_hyperparam()
            array([0])

        The above value is the mean of the distribution.
        """

        hyperparam_guess = numpy.zeros_like(self.shape)

        for i in range(hyperparam_guess.size):
            # Mean of distribution
            mean = self.shape[i] / self.rate[i]
            hyperparam_guess[i] = mean

        return hyperparam_guess

    # ===========
    # check param
    # ===========

    def _check_param(self, x):
        """
        Checks the input x.
        """

        # Convert input to numpy array
        if numpy.isscalar(x):
            x_ = numpy.array([x], dtype=float)
        elif isinstance(x, list):
            x_ = numpy.array(x, dtype=float)
        elif isinstance(x, numpy.ndarray):
            x_ = x
        else:
            raise TypeError('"x" should be scalar, list, or numpy array.')

        # Match the size of self.rate and self.shape with size of input x
        if x_.size == self.rate.size and x_.size == self.shape.size:
            rate_ = self.rate
            shape_ = self.shape
        elif self.rate.size == 1 and self.shape.size == 1:
            rate_ = numpy.tile(self.rate, x_.size)
            shape_ = numpy.tile(self.shape, x_.size)
        else:
            raise ValueError('Size of "x" and "self.rate" or "self.shape" ' +
                             'do not match.')

        return x_, shape_, rate_

    # ===
    # pdf
    # ===

    def pdf(self, x):
        """
        Probability density function of the prior distribution.

        Parameters
        ----------

        x : float or array_like[float]
            Input hyperparameter or an array of hyperparameters.

        Returns
        -------

        pdf : float or array_like[float]
            The probability density function of the input hyperparameter(s).

        See Also
        --------

        :meth:`glearn.priors.Prior.log_pdf`
        :meth:`glearn.priors.Erlang.pdf_jacobian`
        :meth:`glearn.priors.Erlang.pdf_hessian`

        Notes
        -----

        The probability density function is

        .. math::

            p(\\theta \\vert \\alpha, \\beta) = \\frac{\\theta^{\\alpha-1}
            e^{-\\beta \\theta} \\beta^{\\alpha}}{(\\alpha -1)!}.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the Erlang distribution with the shape parameter
        :math:`\\alpha=2` and rate parameter :math:`\\beta=4`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Erlang(2, 4)

            >>> # Evaluate PDF function at multiple locations
            >>> t = [0, 0.5, 1]
            >>> prior.pdf(t)
            array([0.        , 1.08268227, 0.29305022])
        """

        # Convert x or self.rate to arrays of the same size
        x, shape, rate = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):
            coeff = rate[i]**shape[i] / gamma(shape[i])
            a = shape[i] - 1.0
            b = rate[i] * x[i]
            k = numpy.exp(-b)
            m = x[i]**a
            pdf_[i] = coeff * m * k

        return pdf_

    # ============
    # pdf jacobian
    # ============

    def pdf_jacobian(self, x):
        """
        Jacobian of the probability density function of the prior distribution.

        Parameters
        ----------

        x : float or array_like[float]
            Input hyperparameter or an array of hyperparameters.

        Returns
        -------

        jac : float or array_like[float]
            The Jacobian of the probability density function of the input
            hyperparameter(s).

        See Also
        --------

        :meth:`glearn.priors.Prior.log_pdf_jacobian`
        :meth:`glearn.priors.Erlang.pdf`
        :meth:`glearn.priors.Erlang.pdf_hessian`

        Notes
        -----

        The first derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}}{\\mathrm{d}\\theta}
            p(\\theta \\vert \\alpha, \\beta) = \\frac{\\theta^{\\alpha-1}
            e^{-\\beta \\theta} \\beta^{\\alpha}}{(\\alpha - 1)!}
            \\frac{\\alpha-1 -\\beta \\theta}{\\theta},

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the Erlang distribution with the shape parameter
        :math:`\\alpha=2` and rate parameter :math:`\\beta=4`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Erlang(2, 4)

            >>> # Evaluate the Jacobian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_jacobian(t)
            array([        nan, -2.16536453, -0.87915067])
        """

        # Convert x or self.rate to arrays of the same size
        x, shape, rate = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

        for i in range(x.size):
            coeff = rate[i]**shape[i] / gamma(shape[i])
            a = shape[i] - 1.0
            b = rate[i] * x[i]
            k = numpy.exp(-b)
            m = x[i]**a
            pdf_jacobian_[i] = coeff * m * k * (a - b) / x[i]

        return pdf_jacobian_

    # ===========
    # pdf hessian
    # ===========

    def pdf_hessian(self, x):
        """
        Hessian of the probability density function of the prior distribution.

        Parameters
        ----------

        x : float or array_like[float]
            Input hyperparameter or an array of hyperparameters.

        Returns
        -------

        hess : float or array_like[float]
            The Hessian of the probability density function of the input
            hyperparameter(s).

        See Also
        --------

        :meth:`glearn.priors.Prior.log_pdf_hessian`
        :meth:`glearn.priors.Erlang.pdf`
        :meth:`glearn.priors.Erlang.pdf_jacobian`

        Notes
        -----

        The second derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}^2}{\\mathrm{d}\\theta^2}
            p(\\theta \\vert \\alpha, \\beta) = \\frac{\\theta^{\\alpha-1}
            e^{-\\beta \\theta} \\beta^{\\alpha}}{(\\alpha - 1)!}
            \\frac{(\\alpha-1 -\\beta \\theta)^2 - (\\alpha-1)}{\\theta^2},

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the Erlang distribution with the shape parameter
        :math:`\\alpha=2` and rate parameter :math:`\\beta=4`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Erlang(2, 4)

            >>> # Evaluate the Hessian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_hessian(t)
            array([[       nan, 0.        , 0.        ],
                   [0.        , 0.        , 0.        ],
                   [0.        , 0.        , 2.34440178]])
        """

        # Convert x or self.rate to arrays of the same size
        x, shape, rate = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        for i in range(x.size):
            coeff = rate[i]**shape[i] / gamma(shape[i])
            a = shape[i] - 1.0
            b = rate[i] * x[i]
            k = numpy.exp(-b)
            m = x[i]**a
            pdf_hessian_[i, i] = coeff * m * k * ((a - b)**2 - a) / x[i]**2

        return pdf_hessian_
