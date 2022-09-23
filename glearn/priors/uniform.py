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


# =======
# Uniform
# =======

class Uniform(Prior):
    """
    Uniform distribution.

    .. note::

        For the methods of this class, see the base class
        :class:`glearn.priors.Prior`.

    Parameters
    ----------

    a : float or array_like[float], default=0
        The left point of an interval :math:`[a, b]` of the uniform
        distribution. If ``a`` is given as an array :math:`(a_1, \\dots, a_p)`,
        the prior is assumed to be :math:`p` independent distributions, each on
        the interval :math:`[a_i, b_i]`.

    b : float or array_like[float], default=numpy.inf
        The right point of an interval :math:`[a, b]` of the uniform
        distribution. If ``b`` is given as an array :math:`(b_1, \\dots, b_p)`,
        the prior is assumed to be :math:`p` independent distributions, each on
        the interval :math:`[a_i, b_i]`.

    Attributes
    ----------

    a : float or array_like[float], default=0
        Input argument a

    b : float or array_like[float], default=0
        Input argument b

    mean : float or array_like[float], default=0
        Mean of the distribution

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

    The uniform distribution in the interval :math:`[a, b]` is defined by the
    probability density function

    .. math::

        p(\\theta) =
        \\begin{cases}
            1, & a \\leq \\theta \\leq b, \\\\
            0, & \\text{otherwise}.
        \\end{cases}

    **Multiple Hyperparameters:**

    If an array of the hyperparameters are given, namely
    :math:`\\boldsymbol{\\theta} = (\\theta_1, \\dots, \\theta_p)`, then
    the prior is the product of independent priors

    .. math::

        p(\\boldsymbol{\\theta}) = p(\\theta_1) \\dots p(\\theta_p).

    In this case, if the input arguments ``a`` and ``b`` are given as the
    arrays :math:`\\boldsymbol{a} = (a_1, \\dots, a_p)` and
    :math:`\\boldsymbol{b} = (b_1, \\dots, b_p)`, each prior
    :math:`p(\\theta_i)` is defined as the uniform distribution on the interval
    :math:`[a_i, b_i]`. In contrary, if ``a`` and ``b`` are given as the
    scalars :math:`a` and :math:`b`, then all priors :math:`p(\\theta_i)` are
    defined as uniform distributions in the interval :math:`[a, b]`.

    Examples
    --------

    **Create Prior Objects:**

    Create uniform prior in the interval :math:`[0.2, 0.9]`:

    .. code-block:: python

        >>> from glearn import priors
        >>> prior = priors.Uniform(0.2, 0.9)

        >>> # Evaluate PDF function at multiple locations
        >>> t = [0, 0.5, 1]
        >>> prior.pdf(t)
        array([0.        , 1.42857143, 0.        ])

        >>> # Evaluate the Jacobian of the PDF
        >>> prior.pdf_jacobian(t)
        array([0., 0., 0.])

        >>> # Evaluate the Hessian of the PDF
        >>> prior.pdf_hessian(t)
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])

        >>> # Evaluate the log-PDF
        >>> prior.log_pdf(t)
        -inf

        >>> # Evaluate the Jacobian of the log-PDF
        >>> prior.log_pdf_jacobian(t)
        array([nan, nan, nan])

        >>> # Evaluate the Hessian of the log-PDF
        >>> prior.log_pdf_hessian(t)
        array([[nan,  0.,  0.],
               [ 0., nan,  0.],
               [ 0.,  0., nan]])

        >>> # Plot the distribution and its first and second derivative
        >>> prior.plot()

    .. image:: ../_static/images/plots/prior_uniform.png
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

    def __init__(self, a=0, b=numpy.inf):
        """
        Initialization.
        """

        # This distribution does not have half distribution
        half = False
        super().__init__(half)

        # Check arguments
        self.a, self.b = self._check_arguments(a, b)

        # Mean of distribution (could be used for initial hyperparam guess)
        self.mean = self._pdf_mean()

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, a, b):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check type
        if numpy.isscalar(a) and not isinstance(a, (int, float)):
            raise TypeError('"a" should be a float number.')
        if numpy.isscalar(b) and not isinstance(b, (int, float)):
            raise TypeError('"b" should be a float number.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(a):
            a = numpy.array([a], dtype=float)
        elif isinstance(a, list):
            a = numpy.array(a, dtype=float)
        elif not isinstance(a, numpy.ndarray):
            raise TypeError('"a" should be a scalar, list of numpy array.')

        if numpy.isscalar(b):
            b = numpy.array([b], dtype=float)
        elif isinstance(b, list):
            b = numpy.array(b, dtype=float)
        elif not isinstance(b, numpy.ndarray):
            raise TypeError('"b" should be a scalar, list of numpy array.')

        # Check size of a and b
        if a.size != b.size:
            raise ValueError('Sizes of "a" and "b" do not match.')

        # Each element of "a" cannot be larger than the corresponding element
        # of "b"
        if any(a > b):
            raise ValueError('"a" cannot be larger than "b".')

        return a, b

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

        For the uniform distribution in the interval :math:`[a, b]`, the
        suggested hyperparameter is the mid-point of the interval,
        :math:`\\theta = (a+b)/2`.

        If the input arguments ``a`` and ``b`` are given as the arrays
        :math:`\\boldsymbol{a} = (a_1, \\dots, a_p)` and
        :math:`\\boldsymbol{b} = (b_1, \\dots, b_p)`, the suggested array of
        hyperparameters :math:`\\boldsymbol{\\theta} = (\\theta_1, \\dots,
        \\theta_p)` are :math:`\\theta_i = (a_i + b_i) / 2`.

        The suggested hyperparameters can be used as initial guess for the
        optimization of the posterior functions when used with this prior.

        Examples
        --------

        Create uniform prior in the interval :math:`[0.2, 0.9]`:

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Uniform(0.2, 0.9)

            >>> # Find a feasible hyperparameter value
            >>> prior.suggest_hyperparam()
            array([0.55])

        The above value is the mid-point of the interval :math:`[0.2, 0.9]`.
        """

        hyperparam_guess = numpy.zeros_like(self.a)

        for i in range(hyperparam_guess.size):

            if not numpy.isinf(numpy.abs(self.a[i])) and \
                    not numpy.isinf(self.b[i]):
                mean = 0.5 * (self.a[i] + self.b[i])

                if positive and mean <= 0.0:
                    if self.b[i] > 0.0:
                        hyperparam_guess[i] = self.b[i] / 2.0
                    else:
                        raise ValueError('Cannot select a positive ' +
                                         'hyperparameter from the prior ' +
                                         'distribution.')
                else:
                    hyperparam_guess[i] = mean

            elif numpy.isinf(numpy.abs(self.a[i])) and \
                    not numpy.isinf(self.b[i]):
                if positive and self.b[i] <= 0.0:
                    raise ValueError('Cannot select a positive ' +
                                     'hyperparameter from the prior ' +
                                     'distribution.')
                else:
                    hyperparam_guess[i] = self.b[i] / 2.0

            elif not numpy.isinf(numpy.abs(self.a[i])) and \
                    numpy.isinf(self.b[i]):
                if positive and self.a[i] <= 0.0:
                    hyperparam_guess[i] = 1.0
                else:
                    hyperparam_guess[i] = 2.0 * self.a[i]

            else:
                # a and b are infinity. Just pick any finite number.
                hyperparam_guess[i] = 1.0

        return hyperparam_guess

    # =======
    # check x
    # =======

    def _check_param(self, x):
        """
        If x is an array, the prior is the sum of log-priors for each
        x element.
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

        # Match the size of self.a and self.b with size of input x
        if x_.size == self.a.size:
            a_ = self.a
            b_ = self.b
        elif self.a.size == 1:
            a_ = numpy.tile(self.a, x_.size)
            b_ = numpy.tile(self.b, x_.size)
        else:
            raise ValueError('Size of "x" and "self.a" do not match.')

        return x_, a_, b_

    # ========
    # pdf mean
    # ========

    def _pdf_mean(self):
        """
        Returns the mean of pdf.
        """

        if numpy.isinf(numpy.abs(self.a)).any() or numpy.isinf(self.b).any():
            self.mean = numpy.zeros_like(self.a)
            self.mean[:] = numpy.nan
        else:
            self.mean = 0.5 * (self.b - self.a)

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
        :meth:`glearn.priors.Uniform.pdf_jacobian`
        :meth:`glearn.priors.Uniform.pdf_hessian`

        Notes
        -----

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create uniform prior in the interval :math:`[0.2, 0.9]`:

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Uniform(0.2, 0.9)

            >>> # Evaluate PDF function at multiple locations
            >>> t = [0, 0.5, 1]
            >>> prior.pdf(t)
            array([0.        , 1.42857143, 0.        ])
        """

        # Convert hyperparam or self.a, and self.b to arrays of the same size
        x, a, b = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):

            if x[i] > b[i] or x[i] < a[i]:
                # Prior is zero (log prior is -inf) outside [a, b]
                pdf_[i] = 0.0

            elif numpy.isinf(b[i]) or numpy.isinf(numpy.abs(a[i])):
                # Improper prior is 1 (it log is 0) for semi-infinite intervals
                pdf_[i] = 1.0
            else:
                # Uniform proper prior between interval [a, b]
                pdf_[i] = 1.0 / (b[i] - a[i])

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
        :meth:`glearn.priors.Uniform.pdf`
        :meth:`glearn.priors.Uniform.pdf_hessian`

        Notes
        -----

        The first derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}p(\\theta)}{\\mathrm{d}\\theta} = 0.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create uniform prior in the interval :math:`[0.2, 0.9]`:

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Uniform(0.2, 0.9)

            >>> # Evaluate the Jacobian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_jacobian(t)
            array([0., 0., 0.])
        """

        # Convert hyperparam or self.a, and self.b to arrays of the same size
        x, _, _ = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

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
        :meth:`glearn.priors.Uniform.pdf`
        :meth:`glearn.priors.Uniform.pdf_jacobian`

        Notes
        -----

        The second derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}^2p(\\theta)}{\\mathrm{d}\\theta^2} = 0.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create uniform prior in the interval :math:`[0.2, 0.9]`:

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.Uniform(0.2, 0.9)

            >>> # Evaluate the Hessian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_hessian(t)
            array([[0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.]])
        """

        # Convert hyperparam or self.a, and self.b to arrays of the same size
        x, _, _ = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        return pdf_hessian_
