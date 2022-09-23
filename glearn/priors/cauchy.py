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
from .prior import Prior

__all__ = ['Cauchy']


# ======
# Cauchy
# ======

class Cauchy(Prior):
    """
    Cauchy distribution.

    .. note::

        For the methods of this class, see the base class
        :class:`glearn.priors.Prior`.

    Parameters
    ----------

    median : float or array_like[float], default=0.0
        The median :math:`\\theta_0` of Cauchy distribution. If an array
        :math:`\\boldsymbol{\\theta}_0 = (\\theta_{01}, \\dots, \\theta_{0p})`
        is given, the prior is assumed to be :math:`p` independent Cauchy
        distributions each with median :math:`\\theta_{i0}`.

    std : float or array_like[float], default=1.0
        The scale parameter :math:`\\gamma` of Cauchy distribution. If an
        array :math:`\\boldsymbol{\\gamma} = (\\gamma_1, \\dots, \\gamma_p)` is
        given, the prior is assumed to be :math:`p` independent Cauchy
        distributions each with scale :math:`\\gamma_i`.

    half : bool, default=False
        If `True`, the prior is the half-Cauchy distribution.

    Attributes
    ----------

    median : float or array_like[float], default=0
        Median :math:`\\theta_0` of the distribution

    scale : float or array_like[float], default=0
        Scale parameter :math:`\\gamma` of the distribution

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

    The Cauchy distribution with median :math:`\\theta_0` and scale
    :math:`\\gamma` is defined by the probability density function

    .. math::

        p(\\theta \\vert \\theta_0, \\gamma) = \\frac{1}{\\pi \\gamma}
        \\frac{1}{1 + \\left( \\frac{\\theta - \\theta_0}{\\gamma}
        \\right)^2}.

    If ``half`` is `True`, the prior is the half-normal distribution

    .. math::

        p(\\theta \\vert \\theta_0, \\gamma) = \\frac{2}{\\pi \\gamma}
        \\frac{1}{1 + \\left( \\frac{\\theta - \\theta_0}{\\gamma}
        \\right)^2}.

    **Multiple Hyperparameters:**

    If an array of the hyperparameters are given, namely
    :math:`\\boldsymbol{\\theta} = (\\theta_1, \\dots, \\theta_p)`, then
    the prior is the product of independent priors

    .. math::

        p(\\boldsymbol{\\theta}) = p(\\theta_1) \\dots p(\\theta_p).

    In this case, if the input arguments ``median`` and ``scale`` are given as
    the arrays :math:`\\boldsymbol{\\theta}_0 = (\\theta_{01}, \\dots,
    \\theta_{0p})` and :math:`\\boldsymbol{\\gamma} = (\\gamma_1, \\dots,
    \\gamma_p)`, each prior :math:`p(\\theta_i)` is defined as the Cauchy
    distribution with median :math:`\\theta_{0i}` and scale :math:`\\gamma_i`.
    In contrary, if ``median`` and ``scale`` are given as the scalars
    :math:`\\theta_0` and :math:`\\gamma`, then all priors :math:`p(\\theta_i)`
    are defined as the Cauchy distribution with median :math:`\\theta_0` and
    scale :math:`\\gamma`.

    Examples
    --------

    **Create Prior Objects:**

    Create the Cauchy distribution with median :math:`\\theta_0 = 1` and scale
    :math:`\\gamma=2`.

    .. code-block:: python

        >>> from glearn import priors
        >>> prior = priors.Cauchy(1, 2)

        >>> # Evaluate PDF function at multiple locations
        >>> t = [0, 0.5, 1]
        >>> prior.pdf(t)
        array([0.12732395, 0.14979289, 0.15915494])

        >>> # Evaluate the Jacobian of the PDF
        >>> prior.pdf_jacobian(t)
        array([-0.        , -0.03524539, -0.07957747])

        >>> # Evaluate the Hessian of the PDF
        >>> prior.pdf_hessian(t)
        array([[-0.05092958,  0.        ,  0.        ],
               [ 0.        , -0.05390471,  0.        ],
               [ 0.        ,  0.        ,  0.        ]])

        >>> # Evaluate the log-PDF
        >>> prior.log_pdf(t)
        8.651043137341349

        >>> # Evaluate the Jacobian of the log-PDF
        >>> prior.log_pdf_jacobian(t)
        array([-1.15129255, -5.30828143, -5.41784728])

        >>> # Evaluate the Hessian of the log-PDF
        >>> prior.log_pdf_hessian(t)
        array([[-3.97642358,  0.        ,  0.        ],
               [ 0.        ,  3.73231234,  0.        ],
               [ 0.        ,  0.        ,  4.40296037]])

        >>> # Plot the distribution and its first and second derivative
        >>> prior.plot()

    .. image:: ../_static/images/plots/prior_cauchy.png
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

    def __init__(self, median=0.0, scale=1.0, half=False):
        """
        Initialization.
        """

        super().__init__(half)

        # Check arguments
        self.median, self.scale = self._check_arguments(median, scale)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, median, scale):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check type
        if numpy.isscalar(median) and not isinstance(median, (int, float)):
            raise TypeError('"median" should be a float number.')

        if numpy.isscalar(scale) and not isinstance(scale, (int, float)):
            raise TypeError('"scale" should be a float number.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(median):
            median = numpy.array([median], dtype=float)
        elif isinstance(median, list):
            median = numpy.array(median, dtype=float)
        elif not isinstance(median, numpy.ndarray):
            raise TypeError('"median" should be a scalar, list of numpy ' +
                            'array.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(scale):
            scale = numpy.array([scale], dtype=float)
        elif isinstance(scale, list):
            scale = numpy.array(scale, dtype=float)
        elif not isinstance(scale, numpy.ndarray):
            raise TypeError('"scale" should be a scalar, list of numpy array.')

        if any(scale <= 0.0):
            raise ValueError('"scale" should be positive.')

        return median, scale

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

        For the Cauchy distribution, the suggested hyperparameter is the median
        :math:`\\theta_0`. For the half-Cauchy distribution, the suggested
        hyperparameter is the scale :math:`\\gamma`.

        The suggested hyperparameters can be used as initial guess for the
        optimization of the posterior functions when used with this prior.

        Examples
        --------

        Create the Cauchy distribution with median :math:`\\theta_0 = 1` and
        scale :math:`\\gamma=2`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Cauchy(1, 2)

            >>> # Find a feasible hyperparameter value
            >>> prior.suggest_hyperparam()
            array([1.])

        The above value is the mean of the distribution.
        """

        # Median of distribution (could be used for initial hyperparam guess)
        if self.half:
            hyperparam_guess = self.scale
        else:
            if positive and self.median <= 0:
                hyperparam_guess = self.scale
            else:
                hyperparam_guess = self.median

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
            raise TypeError('"x" should be scalar, list, or numpy ' +
                            'array.')

        # Match the size of self.scale and self.median with size of input x
        if x_.size == self.scale.size and x_.size == self.median.size:
            scale_ = self.scale
            median_ = self.median
        elif self.scale.size == 1 and self.median.size == 1:
            scale_ = numpy.tile(self.scale, x_.size)
            median_ = numpy.tile(self.median, x_.size)
        else:
            raise ValueError('Size of "x" and "self.scale" or "self.median" ' +
                             'do not match.')

        return x_, median_, scale_

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
        :meth:`glearn.priors.Cauchy.pdf_jacobian`
        :meth:`glearn.priors.Cauchy.pdf_hessian`

        Notes
        -----

        The probability density function is

        .. math::

            p(\\theta \\vert \\theta_0, \\gamma) = \\frac{1}{\\pi \\gamma}
            \\frac{1}{z}.

        where

        .. math::

            z = 1 + \\left( \\frac{\\theta - \\theta_0}{\\gamma}\\right)^2.

        If ``half`` is `True`, the above function is doubled.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the Cauchy distribution with median :math:`\\theta_0 = 1` and
        scale :math:`\\gamma=2`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Cauchy(1, 2)

            >>> # Evaluate PDF function at multiple locations
            >>> t = [0, 0.5, 1]
            >>> prior.pdf(t)
            array([0.12732395, 0.14979289, 0.15915494])
        """

        # Convert x or self.scale to arrays of the same size
        x, median, scale = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):
            k = 1.0 + ((x[i]-median[i])/scale[i])**2
            pdf_[i] = 1.0 / (scale[i] * numpy.pi * k)

        if self.half:
            pdf_ = 2.0*pdf_

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
        :meth:`glearn.priors.Cauchy.pdf`
        :meth:`glearn.priors.Cauchy.pdf_hessian`

        Notes
        -----

        The first derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}}{\\mathrm{d}\\theta}
            p(\\theta \\vert \\theta_0, \\gamma) = \\frac{1}{\\pi \\gamma^3}
            \\frac{-2\\theta}{z^2}.

        where

        .. math::

            z = 1 + \\left( \\frac{\\theta - \\theta_0}{\\gamma}\\right)^2

        If ``half`` is `True`, the above function is doubled.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the Cauchy distribution with median :math:`\\theta_0 = 1` and
        scale :math:`\\gamma=2`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Cauchy(1, 2)

            >>> # Evaluate the Jacobian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_jacobian(t)
            array([-0.        , -0.03524539, -0.07957747])
        """

        # Convert x or self.scale to arrays of the same size
        x, median, scale = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

        for i in range(x.size):
            k = 1.0 + ((x[i]-median[i])/scale[i])**2
            pdf_jacobian_[i] = -2.0*x[i] / (scale[i]**3 * numpy.pi * k**2)

        if self.half:
            pdf_jacobian_ = 2.0*pdf_jacobian_

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
        :meth:`glearn.priors.Cauchy.pdf`
        :meth:`glearn.priors.Cauchy.pdf_jacobian`

        Notes
        -----

        The second derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}^2}{\\mathrm{d}\\theta^2}
            p(\\theta \\vert \\theta_0, \\gamma) =
            \\frac{8 \\theta^2}{\\pi \\gamma^5 z^3} -
            \\frac{2}{\\pi \\gamma^3 z^2},

        where

        .. math::

            z = 1 + \\left( \\frac{\\theta - \\theta_0}{\\gamma}\\right)^2

        If ``half`` is `True`, the above function is doubled.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the Cauchy distribution with median :math:`\\theta_0 = 1` and
        scale :math:`\\gamma=2`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Cauchy(1, 2)

            >>> # Evaluate the Hessian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_hessian(t)
            array([[-0.05092958,  0.        ,  0.        ],
                   [ 0.        , -0.05390471,  0.        ],
                   [ 0.        ,  0.        ,  0.        ]])
        """

        # Convert x or self.scale to arrays of the same size
        x, median, scale = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        for i in range(x.size):
            k = 1.0 + ((x[i]-median[i])/scale[i])**2
            pdf_hessian_[i, i] = 8.0*x[i]**2 / \
                (scale[i]**5 * numpy.pi * k**3) - \
                2.0 / (scale[i]**3 * numpy.pi * k**2)

        if self.half:
            pdf_hessian_ = 2.0*pdf_hessian_

        return pdf_hessian_
