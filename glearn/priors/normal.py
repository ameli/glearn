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

__all__ = ['Normal']


# ======
# Normal
# ======

class Normal(Prior):
    """
    Normal distribution.

    .. note::

        For the methods of this class, see the base class
        :class:`glearn.priors.Prior`.

    Parameters
    ----------

    mean : float or array_like[float], default=0.0
        The mean :math:`\\mu` of normal distribution. If an array
        :math:`\\boldsymbol{\\mu} = (\\mu_1, \\dots, \\mu_p)` is given, the
        prior is assumed to be :math:`p` independent normal distributions each
        with mean :math:`\\mu_i`.

    std : float or array_like[float], default=1.0
        The standard deviation :math:`\\sigma` of normal distribution. If an
        array :math:`\\boldsymbol{\\sigma} = (\\sigma_1, \\dots, \\sigma_p)` is
        given, the prior is assumed to be :math:`p` independent normal
        distributions each with standard deviation :math:`\\sigma_i`.

    half : bool, default=False
        If `True`, the prior is the half-normal distribution.

    Attributes
    ----------

    mean : float or array_like[float], default=0
        Mean of the distribution

    std : float or array_like[float], default=0
        Standard deviation of the distribution

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

    The normal distribution :math:`\\mathcal{N}(\\mu, \\sigma^2)` is defined by
    the probability density function

    .. math::

        p(\\theta \\vert \\mu, \\sigma^2) = \\frac{1}{\\sigma \\sqrt{2 \\pi}}
        e^{-\\frac{1}{2}z^2},

    where

    .. math::

        z = \\frac{\\theta - \\mu}{\\sigma}.

    If ``half`` is `True`, the prior is the half-normal distribution for
    :math:`\\theta \\geq 0` is

    .. math::

        p(\\theta \\vert \\mu, \\sigma^2) =
        \\frac{\\sqrt{2}}{\\sigma \\sqrt{\\pi}} e^{-\\frac{1}{2}z^2},

    **Multiple Hyperparameters:**

    If an array of the hyperparameters are given, namely
    :math:`\\boldsymbol{\\theta} = (\\theta_1, \\dots, \\theta_p)`, then
    the prior is the product of independent priors

    .. math::

        p(\\boldsymbol{\\theta}) = p(\\theta_1) \\dots p(\\theta_p).

    In this case, if the input arguments ``mean`` and ``std`` are given as the
    arrays :math:`\\boldsymbol{\\mu} = (\\mu_1, \\dots, \\mu_p)` and
    :math:`\\boldsymbol{\\sigma} = (\\sigma_1, \\dots, \\sigma_p)`, each prior
    :math:`p(\\theta_i)` is defined as the normal distribution
    :math:`\\mathcal{N}(\\mu_i, \\sigma_i^2)`. In contrary, if ``mean`` and
    ``sigma`` are given as the scalars :math:`\\mu` and :math:`\\sigma`, then
    all priors :math:`p(\\theta_i)` are defined as the normal distribution
    :math:`\\mathcal{N}(\\mu, \\sigma^2)`.

    Examples
    --------

    **Create Prior Objects:**

    Create the normal distribution :math:`\\mathcal{N}(1, 3^2)`:

    .. code-block:: python

        >>> from glearn import priors
        >>> prior = priors.Normal(1, 3)

        >>> # Evaluate PDF function at multiple locations
        >>> t = [0, 0.5, 1]
        >>> prior.pdf(t)
        array([0.12579441, 0.13114657, 0.13298076])

        >>> # Evaluate the Jacobian of the PDF
        >>> prior.pdf_jacobian(t)
        array([ 0.01397716,  0.00728592, -0.        ])

        >>> # Evaluate the Hessian of the PDF
        >>> prior.pdf_hessian(t)
        array([[-0.01242414,  0.        ,  0.        ],
               [ 0.        , -0.01416707,  0.        ],
               [ 0.        ,  0.        , -0.01477564]])

        >>> # Evaluate the log-PDF
        >>> prior.log_pdf(t)
        -10.812399392266304

        >>> # Evaluate the Jacobian of the log-PDF
        >>> prior.log_pdf_jacobian(t)
        array([ -0.        ,  -1.74938195, -23.02585093])

        >>> # Evaluate the Hessian of the log-PDF
        >>> prior.log_pdf_hessian(t)
        array([[  -0.58909979,    0.        ,    0.        ],
               [   0.        ,   -9.9190987 ,    0.        ],
               [   0.        ,    0.        , -111.92896011]])

        >>> # Plot the distribution and its first and second derivative
        >>> prior.plot()

    .. image:: ../_static/images/plots/prior_normal.png
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

    def __init__(self, mean=0.0, std=1.0, half=False):
        """
        Initialization.
        """

        super().__init__(half)

        # Check arguments
        self.mean, self.std = self._check_arguments(mean, std)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, mean, std):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check type
        if numpy.isscalar(mean) and not isinstance(mean, (int, float)):
            raise TypeError('"mean" should be a float number.')

        if numpy.isscalar(std) and not isinstance(std, (int, float)):
            raise TypeError('"std" should be a float number.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(mean):
            mean = numpy.array([mean], dtype=float)
        elif isinstance(mean, list):
            mean = numpy.array(mean, dtype=float)
        elif not isinstance(mean, numpy.ndarray):
            raise TypeError('"mean" should be a scalar, list of numpy ' +
                            'array.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(std):
            std = numpy.array([std], dtype=float)
        elif isinstance(std, list):
            std = numpy.array(std, dtype=float)
        elif not isinstance(std, numpy.ndarray):
            raise TypeError('"std" should be a scalar, list of numpy array.')

        if any(std <= 0.0):
            raise ValueError('"std" should be positive.')

        return mean, std

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

        For the normal distribution :math:`\\mathcal{N}(\\mu, \\sigma^2)`,
        suggested hyperparameter is the mean :math:`\\mu`.

        If the input arguments ``mean`` is given as an
        :math:`\\boldsymbol{\\mu} = (\\mu_1, \\dots, \\mu_p)`, then
        the output of this function is the array :math:`\\boldsymbol{\\mu}`.

        The suggested hyperparameters can be used as initial guess for the
        optimization of the posterior functions when used with this prior.

        Examples
        --------

        Create the normal distribution :math:`\\mathcal{N}(1, 3^2)`:

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Normal(1, 3)

            >>> # Find a feasible hyperparameter value
            >>> prior.suggest_hyperparam()
            array([1.])

        The above value is the mean of the distribution.
        """

        if self.half:
            # For half-normal distribution, use std as initial hyperparam guess
            hyperparam_guess = self.std
        else:
            if positive and self.mean <= 0.0:
                hyperparam_guess = self.std
            else:
                # Otherwise, use its mean.
                hyperparam_guess = self.mean

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

        # Match the size of self.std and self.mean with size of input x
        if x_.size == self.std.size and x_.size == self.mean.size:
            std_ = self.std
            mean_ = self.mean
        elif self.std.size == 1 and self.mean.size == 1:
            std_ = numpy.tile(self.std, x_.size)
            mean_ = numpy.tile(self.mean, x_.size)
        else:
            raise ValueError('Size of "x" and "self.std" or "self.mean" ' +
                             'do not match.')

        return x_, mean_, std_

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
        :meth:`glearn.priors.Normal.pdf_jacobian`
        :meth:`glearn.priors.Normal.pdf_hessian`

        Notes
        -----

        The probability density function is

        .. math::

            p(\\theta \\vert \\mu, \\sigma^2) =
            \\frac{1}{\\sigma \\sqrt{2 \\pi}} e^{-\\frac{1}{2}z^2},

        where

        .. math::

            z = \\frac{\\theta - \\mu}{\\sigma}.

        If ``half`` is `True`, the above function is doubled.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the normal distribution :math:`\\mathcal{N}(1, 3^2)`:

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Normal(1, 3)

            >>> # Evaluate PDF function at multiple locations
            >>> t = [0, 0.5, 1]
            >>> prior.pdf(t)
            array([0.12579441, 0.13114657, 0.13298076])
        """

        # Convert x or self.std to arrays of the same size
        x, mean, std = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):
            coeff = 1.0 / (std[i] * numpy.sqrt(2.0*numpy.pi))
            m = (x[i] - mean[i]) / std[i]
            k = numpy.exp(-0.5*m**2)
            pdf_[i] = coeff * k

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
        :meth:`glearn.priors.Normal.pdf`
        :meth:`glearn.priors.Normal.pdf_hessian`

        Notes
        -----

        The first derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}}{\\mathrm{d}\\theta}
            p(\\theta \\vert \\mu, \\sigma) =
            -\\frac{1}{\\sigma \\sqrt{2 \\pi}} \\frac{z}{\\sigma}
            e^{-\\frac{1}{2}z^2},

        where

        .. math::

            z = \\frac{\\theta - \\mu}{\\sigma}.

        If ``half`` is `True`, the above function is doubled.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the normal distribution :math:`\\mathcal{N}(1, 3^2)`:

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Normal(1, 3)

            >>> # Evaluate the Jacobian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_jacobian(t)
            array([ 0.01397716,  0.00728592, -0.        ])
        """

        # Convert x or self.std to arrays of the same size
        x, mean, std = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

        for i in range(x.size):
            coeff = 1.0 / (std[i] * numpy.sqrt(2.0*numpy.pi))
            m = (x[i] - mean[i]) / std[i]
            k = numpy.exp(-0.5*m**2)
            pdf_jacobian_[i] = -coeff * m * k / std[i]

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
        :meth:`glearn.priors.Normal.pdf`
        :meth:`glearn.priors.Normal.pdf_jacobian`

        Notes
        -----

        The second derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}^2}{\\mathrm{d}\\theta^2}
            p(\\theta \\vert \\mu, \\sigma) =
            \\frac{1}{\\sigma \\sqrt{2 \\pi}} \\frac{z^2-1}{\\sigma^2}
            e^{-\\frac{1}{2}z^2},

        where

        .. math::

            z = \\frac{\\theta - \\mu}{\\sigma}.

        If ``half`` is `True`, the above function is doubled.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the normal distribution :math:`\\mathcal{N}(1, 3^2)`:

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Normal(1, 3)

            >>> # Evaluate the Hessian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_hessian(t)
            array([[-0.01242414,  0.        ,  0.        ],
                   [ 0.        , -0.01416707,  0.        ],
                   [ 0.        ,  0.        , -0.01477564]])
        """

        # Convert x or self.std to arrays of the same size
        x, mean, std = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        for i in range(x.size):
            coeff = 1.0 / (std[i] * numpy.sqrt(2.0*numpy.pi))
            m = (x[i] - mean[i]) / std[i]
            k = numpy.exp(-0.5*m**2)
            pdf_hessian_[i, i] = coeff * k * (m**2 - 1.0) / std[i]**2

        if self.half:
            pdf_hessian_ = 2.0*pdf_hessian_

        return pdf_hessian_
