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

__all__ = ['InverseGamma']


# =============
# Inverse Gamma
# =============

class InverseGamma(Prior):
    """
    Inverse Gamma distribution.

    .. note::

        For the methods of this class, see the base class
        :class:`glearn.priors.Prior`.

    Parameters
    ----------

    shape : float or array_like[float], default=1.0
        The shape parameter :math:`\\alpha` of Gamma distribution. If an array
        :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_p)` is
        given, the prior is assumed to be :math:`p` independent Gamma
        distributions each with shape :math:`\\alpha_i`.

    rate : float or array_like[float], default=1.0
        The rate :math:`\\beta` of Gamma distribution. If an array
        :math:`\\boldsymbol{\\beta} = (\\beta_1, \\dots, \\beta_p)` is given,
        the prior is assumed to be :math:`p` independent Gamma distributions
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

    The inverse Gamma distribution with shape parameter :math:`\\alpha > 0` and
    rate parameter :math:`\\beta > 0` is defined by the probability density
    function

    .. math::

        p(\\theta \\vert \\alpha, \\beta) =
        \\frac{\\beta^{\\alpha}}{\\Gamma{\\alpha}} \\theta^{-(\\alpha+1)}
        e^{-\\frac{\\beta}{\\theta}},

    where :math:`\\Gamma` is the Gamma function.

    **Multiple Hyperparameters:**

    If an array of the hyperparameters are given, namely
    :math:`\\boldsymbol{\\theta} = (\\theta_1, \\dots, \\theta_p)`, then
    the prior is the product of independent priors

    .. math::

        p(\\boldsymbol{\\theta}) = p(\\theta_1) \\dots p(\\theta_p).

    In this case, if the input arguments ``shape`` and ``rate`` are given as
    the arrays :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_p)`
    and :math:`\\boldsymbol{\\beta} = (\\beta_1, \\dots, \\beta_p)`, each prior
    :math:`p(\\theta_i)` is defined as the inverse Gamma distribution with
    shape parameter :math:`\\alpha_i` and rate parameter :math:`\\beta_i`. In
    contrary, if ``shape`` and ``rate`` are given as the scalars
    :math:`\\alpha` and :math:`\\beta`, then all priors :math:`p(\\theta_i)`
    are defined as the inverse Gamma distribution with shape parameter
    :math:`\\alpha` and rate parameter :math:`\\beta`.

    Examples
    --------

    **Create Prior Objects:**

    Create the inverse Gamma distribution with the shape parameter
    :math:`\\alpha=4` and rate parameter :math:`\\beta=2`.

    .. code-block:: python

        >>> from glearn import priors
        >>> prior = priors.InverseGamma(4, 2)

        >>> # Evaluate PDF function at multiple locations
        >>> t = [0, 0.5, 1]
        >>> prior.pdf(t)
        array([       nan, 1.56293452, 0.36089409])

        >>> # Evaluate the Jacobian of the PDF
        >>> prior.pdf_jacobian(t)
        array([        nan, -3.12586904, -1.08268227])

        >>> # Evaluate the Hessian of the PDF
        >>> prior.pdf_hessian(t)
        array([[         nan,   0.        ,   0.        ],
               [  0.        , -12.50347615,   0.        ],
               [  0.        ,   0.        ,   3.60894089]])

        >>> # Evaluate the log-PDF
        >>> prior.log_pdf(t)
        -17.15935597045384

        >>> # Evaluate the Jacobian of the log-PDF
        >>> prior.log_pdf_jacobian(t)
        array([ -6.90775528, -10.05664278, -11.05240845])

        >>> # Evaluate the Hessian of the log-PDF
        >>> prior.log_pdf_hessian(t)
        array([[-10.60379622,   0.        ,   0.        ],
               [  0.        ,  -3.35321479,   0.        ],
               [  0.        ,   0.        ,  -1.06037962]])

        >>> # Plot the distribution and its first and second derivative
        >>> prior.plot()

    .. image:: ../_static/images/plots/prior_inverse_gamma.png
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

    def __init__(self, shape=1.0, scale=1.0):
        """
        Initialization.
        """

        # This distribution does not have half distribution
        half = False
        super().__init__(half)

        # Check arguments
        self.shape, self.scale = self._check_arguments(shape, scale)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, shape, scale):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check type
        if numpy.isscalar(shape) and not isinstance(shape, (int, float)):
            raise TypeError('"shape" should be a float number.')

        if numpy.isscalar(scale) and not isinstance(scale, (int, float)):
            raise TypeError('"scale" should be a float number.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(shape):
            shape = numpy.array([shape], dtype=float)
        elif isinstance(shape, list):
            shape = numpy.array(shape, dtype=float)
        elif not isinstance(shape, numpy.ndarray):
            raise TypeError('"shape" should be a scalar, list of numpy ' +
                            'array.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(scale):
            scale = numpy.array([scale], dtype=float)
        elif isinstance(scale, list):
            scale = numpy.array(scale, dtype=float)
        elif not isinstance(scale, numpy.ndarray):
            raise TypeError('"scale" should be a scalar, list of numpy array.')

        if any(shape <= 0.0):
            raise ValueError('"shape" should be positive.')
        if any(scale <= 0.0):
            raise ValueError('"scale" should be positive.')

        return shape, scale

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

        For the inverse Gamma distribution with shape parameter
        :math:`\\alpha` and rate parameter :math:`\\beta`, the suggested
        hyperparameter is the mean :math:`\\mu` of the distribution defined by

        .. math::

            \\mu =
            \\begin{cases}
                \\displaystyle{\\frac{\\alpha}{\\beta - 1}}, & \\alpha > 1,
                \\\\
                \\displaystyle{\\frac{\\alpha}{\\beta + 1}}, & \\alpha < 1.
            \\end{cases}

        The suggested hyperparameters can be used as initial guess for the
        optimization of the posterior functions when used with this prior.

        Examples
        --------

        Create the inverse Gamma distribution with the shape parameter
        :math:`\\alpha=4` and rate parameter :math:`\\beta=2`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.InverseGamma(4, 2)

            >>> # Find a feasible hyperparameter value
            >>> prior.suggest_hyperparam()
            array([0.66666667])

        The above value is the mean of the distribution.
        """

        hyperparam_guess = numpy.zeros_like(self.shape)

        for i in range(hyperparam_guess.size):

            # Mean of distribution (could be used for initial hyperparam guess)
            if self.shape[i] > 1.0:
                mean = self.scale[i] / (self.shape[i] - 1.0)
                hyperparam_guess[i] = mean
            else:
                mode = self.scale[i] / (self.shape[i] + 1.0)
                hyperparam_guess[i] = mode

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

        # Match the size of self.scale and self.shape with size of input x
        if x_.size == self.scale.size and x_.size == self.shape.size:
            scale_ = self.scale
            shape_ = self.shape
        elif self.scale.size == 1 and self.shape.size == 1:
            scale_ = numpy.tile(self.scale, x_.size)
            shape_ = numpy.tile(self.shape, x_.size)
        else:
            raise ValueError('Size of "x" and "self.scale" or "self.shape" ' +
                             'do not match.')

        return x_, shape_, scale_

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
        :meth:`glearn.priors.InverseGamma.pdf_jacobian`
        :meth:`glearn.priors.InverseGamma.pdf_hessian`

        Notes
        -----

        The probability density function is

        .. math::

            p(\\theta \\vert \\alpha, \\beta) =
            \\frac{\\beta^{\\alpha}}{\\Gamma{\\alpha}} \\theta^{-(\\alpha+1)}
            e^{-\\frac{\\beta}{\\theta}},

        where :math:`\\Gamma` is the Gamma function.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the inverse Gamma distribution with the shape parameter
        :math:`\\alpha=4` and rate parameter :math:`\\beta=2`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.InverseGamma(4, 2)

            >>> # Evaluate PDF function at multiple locations
            >>> t = [0, 0.5, 1]
            >>> prior.pdf(t)
            array([       nan, 1.56293452, 0.36089409])
        """

        # Convert x or self.scale to arrays of the same size
        x, shape, scale = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):
            coeff = scale[i]**shape[i] / gamma(shape[i])
            a = shape[i] + 1.0
            b = scale[i] / x[i]
            k = numpy.exp(-b)
            m = (1.0 / x[i])**(a)
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
        :meth:`glearn.priors.InverseGamma.pdf`
        :meth:`glearn.priors.InverseGamma.pdf_hessian`

        Notes
        -----

        The first derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}}{\\mathrm{d}\\theta}
            p(\\theta \\vert \\alpha, \\beta) = \\frac{\\theta^{\\alpha-1}
            e^{-\\beta \\theta} \\beta^{\\alpha}}{\\Gamma(\\alpha)}
            \\frac{\\alpha-1 -\\beta \\theta}{\\theta}
            \\left(-\\frac{\\alpha+1}{\\theta} + \\frac{\\beta}{\\theta^2}
            \\right),

        where :math:`\\Gamma` is the Gamma function.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the inverse Gamma distribution with the shape parameter
        :math:`\\alpha=4` and rate parameter :math:`\\beta=2`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.InverseGamma(4, 2)

            >>> # Evaluate the Jacobian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_jacobian(t)
            array([        nan, -3.12586904, -1.08268227])
        """

        # Convert x or self.scale to arrays of the same size
        x, shape, scale = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

        for i in range(x.size):
            coeff = scale[i]**shape[i] / gamma(shape[i])
            a = shape[i] + 1.0
            b = scale[i] / x[i]
            k = numpy.exp(-b)
            m = (1.0 / x[i])**(a)
            pdf_jacobian_[i] = coeff * m * k * \
                (-(shape[i]+1.0)/x[i] + scale[i]/x[i]**2)

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
        :meth:`glearn.priors.InverseGamma.pdf`
        :meth:`glearn.priors.InverseGamma.pdf_jacobian`

        Notes
        -----

        The second derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}^2}{\\mathrm{d}\\theta^2}
            p(\\theta \\vert \\alpha, \\beta) = \\frac{\\theta^{\\alpha-1}
            e^{-\\beta \\theta} \\beta^{\\alpha}}{\\Gamma(\\alpha)}
            \\frac{\\left(a^2 + a - 2ab - 2b + b^2 \\right)}{\\theta^2},

        where :math:`\\Gamma` is the Gamma function, :math:`a = \\alpha+1`, and
        :math:`b = \\frac{\\beta}{\\theta}`.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the inverse Gamma distribution with the shape parameter
        :math:`\\alpha=4` and rate parameter :math:`\\beta=2`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.InverseGamma(4, 2)

            >>> # Evaluate the Hessian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_hessian(t)
            array([[         nan,   0.        ,   0.        ],
                   [  0.        , -12.50347615,   0.        ],
                   [  0.        ,   0.        ,   3.60894089]])
        """

        # Convert x or self.scale to arrays of the same size
        x, shape, scale = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        for i in range(x.size):
            coeff = scale[i]**shape[i] / gamma(shape[i])
            a = shape[i] + 1.0
            b = scale[i] / x[i]
            k = numpy.exp(-b)
            m = (1.0 / x[i])**(a)
            pdf_hessian_[i, i] = (coeff * m * k / x[i]**2) * \
                (a**2 + a - 2.0*a*b - 2*b + b**2)

        return pdf_hessian_
