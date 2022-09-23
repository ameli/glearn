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
from scipy.special import beta as beta_function
from .prior import Prior

__all__ = ['BetaPrime']


# ==========
# Beta Prime
# ==========

class BetaPrime(Prior):
    """
    Beta Prime distribution.

    .. note::

        For the methods of this class, see the base class
        :class:`glearn.priors.Prior`.

    Parameters
    ----------

    shape : float or array_like[float], default=1.0
        The shape parameter :math:`\\alpha` of beta prime distribution. If an
        array :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_p)` is
        given, the prior is assumed to be :math:`p` independent beta prime
        distributions each with shape :math:`\\alpha_i`.

    rate : float or array_like[float], default=1.0
        The rate :math:`\\beta` of beta prime distribution. If an array
        :math:`\\boldsymbol{\\beta} = (\\beta_1, \\dots, \\beta_p)` is given,
        the prior is assumed to be :math:`p` independent beta prime
        distributions each with rate :math:`\\beta_i`.

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

    The beta prime distribution with shape parameter :math:`\\alpha > 0` and
    rate parameter :math:`\\beta > 0` is defined by the probability density
    function

    .. math::

        p(\\theta \\vert \\alpha, \\beta) =
        \\frac{\\theta^{\\alpha-1} (1+\\theta)^{-(\\alpha+\\beta)}}
        {B(\\alpha, \\beta)},

    where :math:`B` is the Beta function.

    **Multiple Hyperparameters:**

    If an array of the hyperparameters are given, namely
    :math:`\\boldsymbol{\\theta} = (\\theta_1, \\dots, \\theta_p)`, then
    the prior is the product of independent priors

    .. math::

        p(\\boldsymbol{\\theta}) = p(\\theta_1) \\dots p(\\theta_p).

    In this case, if the input arguments ``shape`` and ``rate`` are given as
    the arrays :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_p)`
    and :math:`\\boldsymbol{\\beta} = (\\beta_1, \\dots, \\beta_p)`, each prior
    :math:`p(\\theta_i)` is defined as the beta prime distribution with shape
    parameter :math:`\\alpha_i` and rate parameter :math:`\\beta_i`. In
    contrary, if ``shape`` and ``rate`` are given as the scalars
    :math:`\\alpha` and :math:`\\beta`, then all priors :math:`p(\\theta_i)`
    are defined as the beta prime distribution with shape parameter
    :math:`\\alpha`and rate parameter :math:`\\beta`.

    Examples
    --------

    **Create Prior Objects:**

    Create the beta prime distribution with the shape parameter
    :math:`\\alpha=2` and rate parameter :math:`\\beta=4`.

    .. code-block:: python

        >>> from glearn import priors
        >>> prior = priors.BetaPrime(2, 4)

        >>> # Evaluate PDF function at multiple locations
        >>> t = [0, 0.5, 1]
        >>> prior.pdf(t)
        array([0.        , 0.87791495, 0.3125    ])

        >>> # Evaluate the Jacobian of the PDF
        >>> prior.pdf_jacobian(t)
        array([       nan, -1.7558299, -0.625    ])

        >>> # Evaluate the Hessian of the PDF
        >>> prior.pdf_hessian(t)
        array([[       nan, 0.        , 0.        ],
               [0.        , 2.34110654, 0.        ],
               [0.        , 0.        , 1.40625   ]])

        >>> # Evaluate the log-PDF
        >>> prior.log_pdf(t)
        14.661554893429063

        >>> # Evaluate the Jacobian of the log-PDF
        >>> prior.log_pdf_jacobian(t)
        array([ -4.60517019,  -8.19370659, -10.25696996])

        >>> # Evaluate the Hessian of the log-PDF
        >>> prior.log_pdf_hessian(t)
        array([[-7.95284717,  0.        ,  0.        ],
               [ 0.        , -5.80658157,  0.        ],
               [ 0.        ,  0.        , -2.62904039]])

        >>> # Plot the distribution and its first and second derivative
        >>> prior.plot()

    .. image:: ../_static/images/plots/prior_beta_prime.png
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

    def __init__(self, alpha=1.0, beta=1.0):
        """
        Initialization.
        """

        # This distribution does not have half distribution
        half = False
        super().__init__(half)

        # Check arguments
        self.alpha, self.beta = self._check_arguments(alpha, beta)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, alpha, beta):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check type
        if numpy.isscalar(alpha) and not isinstance(alpha, (int, float)):
            raise TypeError('"alpha" should be a float number.')

        if numpy.isscalar(beta) and not isinstance(beta, (int, float)):
            raise TypeError('"beta" should be a float number.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(alpha):
            alpha = numpy.array([alpha], dtype=float)
        elif isinstance(alpha, list):
            alpha = numpy.array(alpha, dtype=float)
        elif not isinstance(alpha, numpy.ndarray):
            raise TypeError('"alpha" should be a scalar, list of numpy ' +
                            'array.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(beta):
            beta = numpy.array([beta], dtype=float)
        elif isinstance(beta, list):
            beta = numpy.array(beta, dtype=float)
        elif not isinstance(beta, numpy.ndarray):
            raise TypeError('"beta" should be a scalar, list of numpy array.')

        if any(alpha <= 0.0):
            raise ValueError('"alpha" should be positive.')
        if any(beta <= 0.0):
            raise ValueError('"beta" should be positive.')

        return alpha, beta

    # ==================
    # suggest hyperparam
    # ==================

    def suggest_hyperparam(self, positive=True):
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

        For the beta prime distribution with shape parameter :math:`\\alpha`
        and rate parameter :math:`\\beta`, the suggested hyperparameter is
        determined as follows:

        * If :math:`\\beta > 1`, the suggested hyperparameter is the mean of
          the distribution

          .. math::

              \\mu = \\frac{\\alpha}{\\beta - 1}.

        * If :math:`\\alpha > 1`, the suggested hyperparameter is the mod of
          the distribution

          .. math::

              \\mu' = \\frac{\\alpha-1}{\\beta + 1}.

        * Other than the above conditions, the number `1` is returned.

        The suggested hyperparameters can be used as initial guess for the
        optimization of the posterior functions when used with this prior.

        Examples
        --------

        Create the beta prime distribution with the shape parameter
        :math:`\\alpha=2` and rate parameter :math:`\\beta=4`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.BetaPrime(2, 4)

            >>> # Find a feasible hyperparameter value
            >>> prior.suggest_hyperparam()
            array([0.6666666666666666])

        The above value is the mean of the distribution.
        """

        hyperparam_guess = numpy.zeros_like(self.alpha)

        for i in range(hyperparam_guess.size):
            # Mean or mode could be used for initial hyperparam guess
            if self.beta[i] > 1.0:
                mean = self.alpha[i] / (self.beta[i] - 1.0)
                hyperparam_guess[i] = mean
            elif self.alpha[i] >= 1.0:

                # Just choose any finite number to avoid infinity.
                if positive and self.alpha[i] == 1.0:
                    hyperparam_guess[i] = 1.0
                else:
                    mode = (self.alpha[i] - 1.0) / (self.beta[i] + 1.0)
                    hyperparam_guess[i] = mode

            else:
                # mean and mode are infinity. Just set any finite number
                hyperparam_guess[i] = 1.0

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

        # Match the size of self.beta and self.alpha with size of input x
        if x_.size == self.beta.size and x_.size == self.alpha.size:
            beta_ = self.beta
            alpha_ = self.alpha
        elif self.beta.size == 1 and self.alpha.size == 1:
            beta_ = numpy.tile(self.beta, x_.size)
            alpha_ = numpy.tile(self.alpha, x_.size)
        else:
            raise ValueError('Size of "x" and "self.beta" or "self.alpha" ' +
                             'do not match.')

        return x_, alpha_, beta_

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
        :meth:`glearn.priors.BetaPrime.pdf_jacobian`
        :meth:`glearn.priors.BetaPrime.pdf_hessian`

        Notes
        -----

        The probability density function is

        .. math::

            p(\\theta \\vert \\alpha, \\beta) =
            \\frac{\\theta^{\\alpha-1} (1+\\theta)^{-(\\alpha+\\beta)}}
            {B(\\alpha, \\beta)},

        where :math:`B` is the Beta function.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the beta prime distribution with the shape parameter
        :math:`\\alpha=2` and rate parameter :math:`\\beta=4`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.BetaPrime(2, 4)

            >>> # Evaluate PDF function at multiple locations
            >>> t = [0, 0.5, 1]
            >>> prior.pdf(t)
            array([0.        , 0.87791495, 0.3125    ])
        """

        # Convert x or self.beta to arrays of the same size
        x, alpha, beta = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):
            coeff = 1.0 / beta_function(alpha[i], beta[i])
            a = alpha[i] - 1.0
            b = -alpha[i] - beta[i]
            k = (1.0 + x[i])**b
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
        :meth:`glearn.priors.BetaPrime.pdf`
        :meth:`glearn.priors.BetaPrime.pdf_hessian`

        Notes
        -----

        The first derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}}{\\mathrm{d}\\theta}
            p(\\theta \\vert \\alpha, \\beta) =
            \\frac{\\theta^{\\alpha-1} (1+\\theta)^{-(\\alpha+\\beta)}}
            {B(\\alpha, \\beta)}
            \\left(\\frac{a}{\\theta} + \\frac{b}{\\theta + 1} \\right),

        where :math:`B` is the Beta function, :math:`a = \\alpha-1`, and
        :math:`b = -(\\alpha + \\beta)`.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the beta prime distribution with the shape parameter
        :math:`\\alpha=2` and rate parameter :math:`\\beta=4`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.BetaPrime(2, 4)

            >>> # Evaluate the Jacobian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_jacobian(t)
            array([       nan, -1.7558299, -0.625    ])
        """

        # Convert x or self.beta to arrays of the same size
        x, alpha, beta = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

        for i in range(x.size):
            coeff = 1.0 / beta_function(alpha[i], beta[i])
            a = alpha[i] - 1.0
            b = -alpha[i] - beta[i]
            k = (1.0 + x[i])**b
            m = x[i]**a
            pdf_jacobian_[i] = coeff * m * k * (a/x[i] + b/(x[i]+1.0))

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
        :meth:`glearn.priors.BetaPrime.pdf`
        :meth:`glearn.priors.BetaPrime.pdf_jacobian`

        Notes
        -----

        The second derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}^2}{\\mathrm{d}\\theta^2}
            p(\\theta \\vert \\alpha, \\beta) =
            \\frac{\\theta^{\\alpha-1} (1+\\theta)^{-(\\alpha+\\beta)}}
            {B(\\alpha, \\beta)}
            \\left(\\frac{a^2}{\\theta^2} -\\frac{a}{\\theta^2} +
            \\frac{2ab}{\\theta (\\theta+1)} + \\frac{b^2}{(\\theta+1)^2} -
            \\frac{b}{(\\theta+1)^2} \\right),

        where :math:`B` is the Beta function, :math:`a = \\alpha-1`, and
        :math:`b = -(\\alpha + \\beta)`.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the beta prime distribution shape parameter :math:`\\alpha=2`
        and rate parameter :math:`\\beta=4`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.BetaPrime(2, 4)

            >>> # Evaluate the Hessian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_hessian(t)
            array([[       nan, 0.        , 0.        ],
                   [0.        , 2.34110654, 0.        ],
                   [0.        , 0.        , 1.40625   ]])
        """

        # Convert x or self.beta to arrays of the same size
        x, alpha, beta = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        for i in range(x.size):
            coeff = 1.0 / beta_function(alpha[i], beta[i])
            a = alpha[i] - 1.0
            b = -alpha[i] - beta[i]
            k = (1.0 + x[i])**b
            m = x[i]**a
            pdf_hessian_[i, i] = coeff * m * k * \
                ((a**2/x[i]**2) - a/x[i]**2 + 2.0*a*b/(x[i]*(x[i]+1.0)) +
                    b**2/((x[i]+1.0)**2) - b/((x[i]+1.0)**2))

        return pdf_hessian_
