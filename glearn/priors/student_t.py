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

__all__ = ['StudentT']


# =========
# Student T
# =========

class StudentT(Prior):
    """
    Student's t-distribution.

    .. note::

        For the methods of this class, see the base class
        :class:`glearn.priors.Prior`.

    Parameters
    ----------

    dof : float or array_like[float], default=1.0
        Degrees of freedom :math:`\\nu` of Student' t-distribution. If an
        array :math:`\\boldsymbol{\\nu} = (\\nu_1, \\dots, \\nu_p)` is given,
        the prior is assumed to be :math:`p` independent Student's
        t-distributions each with degrees of freedom :math:`\\nu_i`.

    half : bool, default=False
        If `True`, the prior is the half-normal distribution.

    Attributes
    ----------

    dof : float or array_like[float], default=0
        Degrees of freedom :math:`\\nu` of the distribution

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

    The Student's t-distribution with degrees of freedom :math:`\\nu` is
    defined by the probability density function

    .. math::

        p(\\theta \\vert \\nu) = \\frac{\\Gamma(\\nu')}{\\sqrt{\\nu \\pi}
        \\Gamma(\\frac{\\nu}{2})} z^{-\\nu'}.

    where :math:`\\nu' = \\frac{1 + \\nu}{2}`, :math:`\\Gamma` is the Gamma
    function, and

    .. math::

        z = 1 + \\frac{\\theta^2}{\\nu}.

    If ``half`` is `True`, the prior is the half Student's t-distribution for
    :math:`\\theta \\geq 0` is

    .. math::

        p(\\theta \\vert \\nu) = 2 \\frac{\\Gamma(\\nu')}{\\sqrt{\\nu \\pi}
        \\Gamma(\\frac{\\nu}{2})} z^{-\\nu'}.

    **Multiple Hyperparameters:**

    If an array of the hyperparameters are given, namely
    :math:`\\boldsymbol{\\theta} = (\\theta_1, \\dots, \\theta_p)`, then
    the prior is the product of independent priors

    .. math::

        p(\\boldsymbol{\\theta}) = p(\\theta_1) \\dots p(\\theta_p).

    In this case, if the input arguments ``dof`` is given as the array
    :math:`\\boldsymbol{\\nu} = (\\nu_1, \\dots, \\nu_p)` each prior
    :math:`p(\\theta_i)` is defined as the Student's t-distribution with the
    degrees of freedom :math:`\\nu_i`. In contrary, if ``dof`` is given by the
    scalar :math:`\\nu`, then all priors :math:`p(\\theta_i)` are defined as
    the Student'st-distribution with degrees of freedom :math:`\\nu`.

    Examples
    --------

    **Create Prior Objects:**

    Create the Student' t-distribution with the degrees of freedom
    :math:`\\nu=4`.

    .. code-block:: python

        >>> from glearn import priors
        >>> prior = priors.StudentT(4)

        >>> # Evaluate PDF function at multiple locations
        >>> t = [0, 0.5, 1]
        >>> prior.pdf(t)
        array([-0.        , -0.18956581, -0.21466253])

        >>> # Evaluate the Jacobian of the PDF
        >>> prior.pdf_jacobian(t)
        array([ 0.01397716,  0.00728592, -0.        ])

        >>> # Evaluate the Hessian of the PDF
        >>> prior.pdf_hessian(t)
        array([[-0.46875   ,  0.        ,  0.        ],
               [ 0.        , -0.22301859,  0.        ],
               [ 0.        ,  0.        ,  0.08586501]])

        >>> # Evaluate the log-PDF
        >>> prior.log_pdf(t)
        14.777495403612827

        >>> # Evaluate the Jacobian of the log-PDF
        >>> prior.log_pdf_jacobian(t)
        array([ -2.30258509,  -8.22351819, -11.07012064])

        >>> # Evaluate the Hessian of the log-PDF
        >>> prior.log_pdf_hessian(t)
        array([[ -8.48303698,   0.        ,   0.        ],
               [  0.        , -10.82020023,   0.        ],
               [  0.        ,   0.        ,  -1.96076114]])

        >>> # Plot the distribution and its first and second derivative
        >>> prior.plot()

    .. image:: ../_static/images/plots/prior_studentt.png
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

    def __init__(self, dof=1.0, half=False):
        """
        Initialization.
        """

        super().__init__(half)

        # Check arguments
        self.dof = self._check_arguments(dof)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, dof):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check type
        if numpy.isscalar(dof) and not isinstance(dof, (int, float)):
            raise TypeError('"dof" should be a float number.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(dof):
            dof = numpy.array([dof], dtype=float)
        elif isinstance(dof, list):
            dof = numpy.array(dof, dtype=float)
        elif not isinstance(dof, numpy.ndarray):
            raise TypeError('"dof" should be a scalar, list of numpy array.')

        if any(dof <= 0.0):
            raise ValueError('"dof" should be positive.')

        return dof

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

        For the Student's t-distribution with the degrees of freedom
        :math:`\\nu`, the suggested hyperparameter is the standard deviation
        of the distribution at :math:`\\nu > 2` is

        .. math::

            \\sigma = \\sqrt{\\frac{\\nu}{\\nu - 2}}.

        If ``half`` is `True`, the stand deviation is

        .. math::

            \\sigma = \\sqrt{\\frac{2\\nu}{\\nu - 2}}.

        .. warning::

            If :math:`\\nu \\leq 2`, the Student's t-distribution does not have
            mean and standard deviation. Hence, this function just returns the
            number `1` to be used as as initial guess.

        The suggested hyperparameters can be used as initial guess for the
        optimization of the posterior functions when used with this prior.

        Examples
        --------

        Create the Student' t-distribution with the degrees of freedom
        :math:`\\nu=4`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.StudentT(4)

            >>> # Find a feasible hyperparameter value
            >>> prior.suggest_hyperparam()
            array([1.41421356])

        The above value is the standard deviation of the distribution.
        """

        hyperparam_guess = numpy.zeros_like(self.dof)

        for i in range(hyperparam_guess.size):

            # std of distribution (could be used for initial hyperparam guess)
            if self.dof[i] > 2.0:
                std = numpy.sqrt(self.dof[i] / (self.dof[i] - 2.0))

                if self.half:
                    std = std * numpy.sqrt(2.0)

                hyperparam_guess[i] = std

            else:
                # mean and std are infinity. Just pick any finite number
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

        # Match the size of self.a and self.b with size of input x
        if x_.size == self.dof.size:
            dof_ = self.dof
        elif self.dof.size == 1:
            dof_ = numpy.tile(self.dof, x_.size)
        else:
            raise ValueError('Size of "x" and "self.a" do not match.')

        return x_, dof_

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
        :meth:`glearn.priors.StudentT.pdf_jacobian`
        :meth:`glearn.priors.StudentT.pdf_hessian`

        Notes
        -----

        The probability density function is

        .. math::

            p(\\theta \\vert \\nu) = \\frac{\\Gamma(\\nu')}{\\sqrt{\\nu \\pi}
            \\Gamma(\\frac{\\nu}{2})} z^{-\\nu'}.

        where :math:`\\nu' = \\frac{1 + \\nu}{2}`, :math:`\\Gamma` is the Gamma
        function, and

        .. math::

            z = 1 + \\frac{\\theta^2}{\\nu}.

        If ``half`` is `True`, the above function is doubled.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the Student' t-distribution with the degrees of freedom
        :math:`\\nu=4`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.StudentT(4)

            >>> # Evaluate PDF function at multiple locations
            >>> t = [0, 0.5, 1]
            >>> prior.pdf(t)
            array([-0.        , -0.18956581, -0.21466253])
        """

        # Convert x or self.dof to arrays of the same size
        x, dof = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):
            ex = 0.5 * (dof[i] + 1.0)
            coeff = gamma(ex) / \
                (numpy.sqrt(dof[i] * numpy.pi) * gamma(0.5*dof[i]))
            k = 1.0 + x[i]**2 / self.dof
            pdf_[i] = coeff * k**(-ex)

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
        :meth:`glearn.priors.StudentT.pdf`
        :meth:`glearn.priors.StudentT.pdf_hessian`

        Notes
        -----

        The first derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}}{\\mathrm{d}\\theta}
            p(\\theta \\vert \\nu) = -\\frac{\\Gamma(\\nu')}{\\sqrt{\\nu \\pi}
            \\Gamma(\\frac{\\nu}{2})} \\nu' z^{-\\nu'-1}
            \\frac{2\\theta}{\\nu},

        where :math:`\\nu' = \\frac{1 + \\nu}{2}`, :math:`\\Gamma` is the Gamma
        function, and

        .. math::

            z = 1 + \\frac{\\theta^2}{\\nu}.

        If ``half`` is `True`, the above function is doubled.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the Student' t-distribution with the degrees of freedom
        :math:`\\nu=4`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.StudentT(4)

            >>> # Evaluate the Jacobian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_jacobian(t)
            array([ 0.01397716,  0.00728592, -0.        ])
        """

        # Convert x or self.dof to arrays of the same size
        x, dof = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

        for i in range(x.size):
            ex = 0.5 * (dof[i] + 1.0)
            coeff = gamma(ex) / \
                (numpy.sqrt(dof[i] * numpy.pi) * gamma(0.5*dof[i]))
            k = 1.0 + x[i]**2 / dof[i]
            pdf_jacobian_[i] = -coeff * ex * k**(-ex-1.0) * (2.0*x[i]/dof[i])

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
        :meth:`glearn.priors.StudentT.pdf`
        :meth:`glearn.priors.StudentT.pdf_jacobian`

        Notes
        -----

        The second derivative of the probability density function is

        .. math::

            \\frac{\\mathrm{d}^2}{\\mathrm{d}\\theta^2}
            p(\\theta \\vert \\nu) =
            -\\frac{\\Gamma(\\nu')}{\\sqrt{\\nu \\pi}
            \\Gamma(\\frac{\\nu}{2})} \\frac{\\nu'}{\\nu}
            \\left( z^{-\\nu'-1} - (\\nu+1) z^{-\\nu'-2}
            \\frac{2 \\theta^2}{\\nu} \\right).

        where :math:`\\nu' = \\frac{1 + \\nu}{2}`, :math:`\\Gamma` is the Gamma
        function, and

        .. math::

            z = 1 + \\frac{\\theta^2}{\\nu}.

        If ``half`` is `True`, the above function is doubled.

        When an array of hyperparameters are given, it is assumed that prior
        for each hyperparameter is independent of others.

        Examples
        --------

        Create the Student' t-distribution with the degrees of freedom
        :math:`\\nu=4`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.StudentT(4)

            >>> # Evaluate the Hessian of the PDF
            >>> t = [0, 0.5, 1]
            >>> prior.pdf_hessian(t)
            array([[-0.46875   ,  0.        ,  0.        ],
                   [ 0.        , -0.22301859,  0.        ],
                   [ 0.        ,  0.        ,  0.08586501]])
        """

        # Convert x or self.dof to arrays of the same size
        x, dof = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        for i in range(x.size):
            ex = 0.5 * (dof[i] + 1.0)
            coeff = gamma(ex) / \
                (numpy.sqrt(dof[i] * numpy.pi) * gamma(0.5*dof[i]))
            k = 1.0 + x[i]**2 / dof[i]
            pdf_hessian_[i, i] = -(2.0 * coeff * ex / dof[i]) * \
                (k**(-ex-1.0) -
                    (ex+1.0) * x[i] * k**(-ex-2.0) * (2.0 * x[i] / dof[i]))

        if self.half:
            pdf_hessian_ = 2.0*pdf_hessian_

        return pdf_hessian_
