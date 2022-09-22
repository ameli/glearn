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
from .._utilities.plot_utilities import *                    # noqa: F401, F403
from .._utilities.plot_utilities import load_plot_settings, plt, \
    save_plot, show_or_save_plot

__all__ = ['Prior']


# =====
# Prior
# =====

class Prior(object):
    """
    Base class for prior distributions.

    .. warning::

        This class is a base class and does not implement a kernel function.
        Use the derivative of this class instead.

    Attributes
    ----------

    use_log_scale : bool, default=True
        If `True`, the input argument to the functions is assumed to be the
        logarithm of the hyperparameter :math:`\\theta`. If `False`, the
        input argument to the functions is assumed to be the hyperparameter
        :math:`\\theta`.

    half : bool, default=False
        If `True`, the probability distribution is assumed to be the
        half-distribution.

    Methods
    -------

    log_pdf
    log_pdf_jacobian
    log_pdf_hessian
    plot

    See Also
    --------

    glearn.priors.Uniform
    glearn.priors.Normal
    glearn.priors.StudentT
    glearn.priors.Cauchy
    glearn.priors.Gamma
    glearn.priors.InverseGamma
    glearn.priors.Erlang
    glearn.priors.BetaPrime

    Examples
    --------

    **Create Prior Objects:**

    Create the inverse Gamma distribution (see
    :class:`glearn.priors.InverseGamma`) with the shape parameter
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

    def __init__(self, half=False):
        """
        Initialization.
        """

        # When True, derivatives of the pdf are taken w.r.t the logarithm of
        # the input hyperparameter. Default is True, but the Posterior class
        # can overwrite this attribute:
        self.use_log_scale = True

        # Using half distribution
        self.half = half

    # ===================
    # scale to hyperparam
    # ===================

    def _scale_to_hyperparam(self, scale):
        """
        Sets hyperparam from scale. ``scale`` is always given with no log-scale
        If self.use_log_eta is True, hyperparam is set as log10 of scale,
        otherwise, just as scale.
        """

        # If log scale is used, output hyperparam is log of scale.
        if self.use_log_scale:
            hyperparam = numpy.log10(numpy.abs(scale))
        else:
            hyperparam = numpy.abs(scale)

        return hyperparam

    # ===================
    # hyperparam to scale
    # ===================

    def _hyperparam_to_scale(self, hyperparam):
        """
        Sets scale from hyperparam. If self.use_log_scale is True, hyperparam
        is the log10 of scale, hence, 10**hyperparam is set to scale. If
        self.use_log_scale is False, hyperparam is directly set to scale.
        """

        # Convert to numpy array
        if numpy.isscalar(hyperparam):
            hyperparam = numpy.array([hyperparam], dtype=float)
        elif isinstance(hyperparam, list):
            hyperparam = numpy.array(hyperparam, dtype=float)

        # If log scale is used, input hyperparam is log of the scale.
        if self.use_log_scale:
            scale = 10.0**hyperparam
        else:
            scale = numpy.abs(hyperparam)

        return scale

    # =======
    # log pdf
    # =======

    def log_pdf(self, hyperparam):
        """
        Logarithm of the probability density function of the prior
        distribution.

        Parameters
        ----------

        x : float or array_like[float]
            Input hyperparameter or an array of hyperparameters.

        Returns
        -------

        pdf : float or array_like[float]
            The logarithm of probability density function of the input
            hyperparameter(s).

        See Also
        --------

        :meth:`glearn.priors.Prior.log_pdf_jacobian`
        :meth:`glearn.priors.Prior.log_pdf_hessian`

        Notes
        -----

        This function returns :math:`\\log p(\\theta)`.

        **Multiple hyperparameters:**

        When an array of hyperparameters :math:`\\boldsymbol{\\theta} =
        (\\theta_, \\dots, \\theta_n)` are given, it is assumed that prior
        for each hyperparameter is independent of others. The output of this
        function is then the sum of all log-probabilities

        .. math::

            \\sum_{i=1}^n \\log p(\\theta_i).

        **Using Log Scale:**

        If the attribute ``use_log_scale`` is `True`, it is assumed that the
        input argument :math:`\\theta` is the log of the hyperparameter, so to
        convert back to the original hyperparameter, the transformation below
        is performed

        .. math::

            \\theta \\gets 10^{\\theta}.

        Examples
        --------

        Create the inverse Gamma distribution with the shape parameter
        :math:`\\alpha=4` and rate parameter :math:`\\beta=2`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.InverseGamma(4, 2)

            >>> # Evaluate the log-PDF
            >>> prior.log_pdf(t)
            -17.15935597045384
        """

        # Convert hyperparam from log to non-log.
        scale = self._hyperparam_to_scale(hyperparam)

        if self.half and any(scale < 0.0):
            raise ValueError('"hyperparam" cannot be negative for ' +
                             'half-distributions.')

        # Call derived class's method
        pdf_ = self.pdf(scale)

        # Take log of the product of all distributions
        log_pdf_ = numpy.sum(numpy.log(pdf_))

        return log_pdf_

    # ================
    # log pdf jacobian
    # ================

    def log_pdf_jacobian(self, hyperparam):
        """
        Jacobian of the logarithm of the probability density function of the
        prior distribution.

        Parameters
        ----------

        x : float or array_like[float]
            Input hyperparameter or an array of hyperparameters.

        Returns
        -------

        jac : float or array_like[float]
            The Jacobian of the logarithm of probability density function of
            the input hyperparameter(s).

        See Also
        --------

        :meth:`glearn.priors.Prior.log_pdf`
        :meth:`glearn.priors.Prior.log_pdf_hessian`

        Notes
        -----

        **Multiple hyperparameters:**

        Given an array of hyperparameters :math:`\\boldsymbol{\\theta} =
        (\\theta_, \\dots, \\theta_n)`, this function returns the Jacobian
        vector :math:`\\boldsymbol{J}` with the components :math:`J_i` as

        .. math::

            J_i= \\frac{\\partial}{\\partial \\theta_i}
            \\log p(\\theta_i) =
            \\frac{1}{p(\\theta_i)}
            \\frac{\\partial p(\\theta_i)}{\\partial \\theta_i}.

        **Using Log Scale:**

        If the attribute ``use_log_scale`` is `True`, it is assumed that the
        input argument :math:`\\theta` is the log of the hyperparameter, so to
        convert back to the original hyperparameter, the transformation below
        is performed

        .. math::

            \\theta \\gets 10^{\\theta}.

        As a result, the Jacobian is transformed by

        .. math::

            J_i \\gets \\log_e(10) \\theta_i J_i.

        Examples
        --------

        Create the inverse Gamma distribution with the shape parameter
        :math:`\\alpha=4` and rate parameter :math:`\\beta=2`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.InverseGamma(4, 2)

            >>> # Evaluate the Jacobian of the log-PDF
            >>> prior.log_pdf_jacobian(t)
            array([ -6.90775528, -10.05664278, -11.05240845])
        """

        # Convert hyperparam from log to non-log (if needed)
        scale = self._hyperparam_to_scale(hyperparam)

        if self.half and any(scale < 0.0):
            raise ValueError('"hyperparam" cannot be negative for ' +
                             'half-distributions.')

        # Call derived class's method
        pdf_ = self.pdf(scale)
        pdf_jacobian_ = self.pdf_jacobian(scale)

        # Take log of the pdf
        log_pdf_jacobian_ = numpy.zeros((scale.size, ), dtype=float)
        for i in range(scale.size):
            log_pdf_jacobian_[i] = pdf_jacobian_[i] / pdf_[i]

        # Convert derivative w.r.t log of scale
        if self.use_log_scale:
            for i in range(scale.size):
                log_pdf_jacobian_[i] = log_pdf_jacobian_[i] * scale[i] * \
                        numpy.log(10.0)

        return log_pdf_jacobian_

    # ===============
    # log pdf hessian
    # ===============

    def log_pdf_hessian(self, hyperparam):
        """
        Hessian of the logarithm of the probability density function of the
        prior distribution.

        Parameters
        ----------

        x : float or array_like[float]
            Input hyperparameter or an array of hyperparameters.

        Returns
        -------

        hess : float or array_like[float]
            The Hessian of the logarithm of probability density function of
            the input hyperparameter(s).

        See Also
        --------

        :meth:`glearn.priors.Prior.log_pdf`
        :meth:`glearn.priors.Prior.log_pdf_jacobian`

        Notes
        -----

        **Multiple hyperparameters:**

        Given an array of hyperparameters :math:`\\boldsymbol{\\theta} =
        (\\theta_, \\dots, \\theta_n)`, this function returns the Jacobian
        vector :math:`\\mathbf{H}` with the components :math:`H_{ij} = 0` if
        :math:`i \\neq j` and

        .. math::

            H_{ii} = \\frac{\\partial^2}{\\partial \\theta_i^2}
            \\log p(\\theta_i) =
            \\frac{1}{p(\\theta_i)}
            \\frac{\\partial^2 p(\\theta_i)}{\\partial \\theta_i^2}
            - \\left( \\frac{J_i}{p(\\theta_i)} \\right)^2,

        where :math:`J_i` is the Jacobian

        .. math::

            J_i = \\frac{\\partial}{\\partial \\theta_i} \\log p(\\theta_i).

        **Using Log Scale:**

        If the attribute ``use_log_scale`` is `True`, it is assumed that the
        input argument :math:`\\theta` is the log of the hyperparameter, so to
        convert back to the original hyperparameter, the transformation below
        is performed

        .. math::

            \\theta \\gets 10^{\\theta}.

        As a result, the Hessian is transformed by

        .. math::

            H_{ij} \\gets
            \\begin{cases}
                H_{ij} \\theta_i^2 (\\log_e(10))^2 + J_i \\log_e(10), & i=j,
                \\\\
                H_{ij} \\theta_i \\theta_j (\\log_e(10))^2, & i \\neq j.
            \\end{cases}

        Examples
        --------

        Create the inverse Gamma distribution with the shape parameter
        :math:`\\alpha=4` and rate parameter :math:`\\beta=2`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.InverseGamma(4, 2)

            >>> # Evaluate the Hessian of the log-PDF
            >>> prior.log_pdf_hessian(t)
            array([[-10.60379622,   0.        ,   0.        ],
                   [  0.        ,  -3.35321479,   0.        ],
                   [  0.        ,   0.        ,  -1.06037962]])
        """

        # Convert hyperparam from log to non-log (if needed)
        scale = self._hyperparam_to_scale(hyperparam)

        if self.half and any(scale < 0.0):
            raise ValueError('"hyperparam" cannot be negative for ' +
                             'half-distributions.')

        # Call derived class's method
        pdf_ = self.pdf(scale)
        pdf_jacobian_ = self.pdf_jacobian(scale)
        pdf_hessian_ = self.pdf_hessian(scale)

        # Take log of the pdf
        log_pdf_hessian_ = numpy.zeros((scale.size, scale.size), dtype=float)
        for i in range(scale.size):
            log_pdf_hessian_[i, i] = (pdf_hessian_[i, i] / pdf_[i]) - \
                    (pdf_jacobian_[i] / pdf_[i])**2

        # Convert derivative w.r.t log of scale
        if self.use_log_scale:

            # To convert derivative to log scale, Jacobian is needed. Note:
            # The Jacobian itself is already converted to log scale.
            log_pdf_jacobian_ = self.log_pdf_jacobian(hyperparam)

            for p in range(scale.size):
                for q in range(scale.size):
                    if p == q:

                        # log_pdf_jacobian_ is already converted to log scale
                        log_pdf_hessian_[p, q] = log_pdf_hessian_[p, q] * \
                            scale[p]**2 * (numpy.log(10.0)**2) + \
                            log_pdf_jacobian_[p] * numpy.log(10.0)
                    else:
                        log_pdf_hessian_[p, q] = log_pdf_hessian_[p, q] * \
                            scale[p] * scale[q] * (numpy.log(10.0)**2)

        return log_pdf_hessian_

    # ====
    # plot
    # ====

    def plot(
            self,
            interval=[0, 2],
            log_scale=False,
            compare_numerical=False,
            test=False):
        """
        Plot the kernel function and its first and second derivative.

        Parameters
        ----------

        interval : float, default=[0, 2]
            The abscissa interval of the plot.

        log_scale : bool, default=False
            If `True`, the hyperparameter (abscissa) is assumed to be in the
            logarithmic scale.

        compare_numerical : bool, default=False
            It `True`, it computes the derivatives of the prior distribution
            and plots the numerical derivatives together with the exact values
            of the derivatives from analytical formula. This is used to
            validate the analytical formulas.


        test : bool, default=False
            If `True`, this function is used for test purposes.

        Notes
        -----

        * If no graphical backend exists (such as running the code on a remote
          server or manually disabling the X11 backend), the plot will not be
          shown, rather, it will be saved as an ``svg`` file in the current
          directory.
        * If the executable ``latex`` is available on ``PATH``, the plot is
          rendered using :math:`\\rm\\LaTeX` and it may take slightly longer to
          produce the plot.
        * If :math:`\\rm\\LaTeX` is not installed, it uses any available
          San-Serif font to render the plot.

        To manually disable interactive plot display and save the plot as
        ``svg`` instead, add the following at the very beginning of your code
        before importing :mod:`glearn`:

        .. code-block:: python

            >>> import os
            >>> os.environ['GLEARN_NO_DISPLAY'] = 'True'

        Examples
        --------

        Create the inverse Gamma distribution with the shape parameter
        :math:`\\alpha=4` and rate parameter :math:`\\beta=2`.

        .. code-block:: python

            >>> from glearn import priors
            >>> prior = priors.InverseGamma(4, 2)

            >>> # Plot the distribution and its first and second derivative
            >>> prior.plot()

        .. image:: ../_static/images/plots/prior_inverse_gamma.png
            :align: center
            :width: 100%
            :class: custom-dark
        """

        load_plot_settings()

        # Check range
        if not isinstance(interval, (list, tuple)):
            raise TypeError('"interval" should be a list or a tuple')
        elif len(interval) != 2:
            raise ValueError('"interval" should be 1d array of size 2.')
        elif interval[0] >= interval[1]:
            raise ValueError('"interval[0]" should be less than ' +
                             '"interval[1]".')

        # Avoid plotting from origin in log-scale x-axis
        if log_scale and interval[0] == 0.0:
            interval[0] = numpy.min([(interval[1] - interval[0]) * 1e-2, 1e-2])

        # Abscissa
        num_points = 200
        if log_scale:
            x = numpy.logspace(numpy.log10(interval[0]),
                               numpy.log10(interval[1]), num_points)
        else:
            x = numpy.linspace(interval[0], interval[1], num_points)

        # Convert x to log of x (if enabled by log_scale)
        if log_scale:
            hyperparam = numpy.log10(numpy.abs(x))
        else:
            # Note: don't use abs(x), for some distributions, x may be negative
            hyperparam = x

        # Allocate outputs
        d0f = numpy.zeros_like(hyperparam)
        d1f = numpy.zeros_like(hyperparam)
        d2f = numpy.zeros_like(hyperparam)

        # Generate distribution and its derivatives
        for i in range(hyperparam.size):

            # Compute the pdf and its first and second derivative
            if log_scale:
                d0f[i] = self.log_pdf(hyperparam[i])
                d1f[i] = self.log_pdf_jacobian(hyperparam[i])
                d2f[i] = self.log_pdf_hessian(hyperparam[i])
            else:
                d0f[i] = self.pdf(hyperparam[i])
                d1f[i] = self.pdf_jacobian(hyperparam[i])
                d2f[i] = self.pdf_hessian(hyperparam[i])

        # Compare analytic derivative with numerical derivative
        if compare_numerical:
            d1f_num = numpy.zeros_like(hyperparam.size-2)
            d2f_num = numpy.zeros_like(hyperparam.size-4)

            d1f_num = (d0f[2:] - d0f[:-2]) / (hyperparam[2:] - hyperparam[:-2])
            d2f_num = (d1f_num[2:] - d1f_num[:-2]) / \
                (hyperparam[3:-1] - hyperparam[1:-3])

        # Plotting
        fig, ax = plt.subplots(ncols=3, figsize=(12.5, 4))
        ax[0].plot(x, d0f, color='black')
        ax[1].plot(x, d1f, color='black', label='analytic')
        ax[2].plot(x, d2f, color='black', label='analytic')
        ax[0].set_xlabel(r'$\theta$')
        ax[1].set_xlabel(r'$\theta$')
        ax[2].set_xlabel(r'$\theta$')

        if compare_numerical:
            ax[1].plot(x[1:-1], d1f_num, '--', color='black',
                       label='numerical')
            ax[2].plot(x[2:-2], d2f_num, '--', color='black',
                       label='numerical')
            ax[1].legend()
            ax[2].legend()

        if log_scale:
            ax[0].set_ylabel(r'$\ln p(\theta)$')
        else:
            ax[0].set_ylabel(r'$p(\theta)$')

        if log_scale:
            ax[1].set_ylabel(r'$\frac{\mathrm{d}\ln p(\theta)}' +
                             r'{\mathrm{d}(\ln \theta)}$')
        else:
            ax[1].set_ylabel(r'$\frac{\mathrm{d}p(\theta)}' +
                             r'{\mathrm{d}\theta}$')

        if log_scale:
            ax[2].set_ylabel(r'$\frac{\mathrm{d}^2\ln p(\theta)}' +
                             r'{\mathrm{d}(\ln \theta)^2}$')
        else:
            ax[2].set_ylabel(r'$\frac{\mathrm{d}^2 p(\theta)}' +
                             r'{\mathrm{d}\theta^2}$')

        ax[0].set_title('Probability Density Function (PDF)')
        ax[1].set_title('First derivative of PDF')
        ax[2].set_title('Second derivative of PDF')
        ax[0].set_xlim([interval[0], interval[1]])
        ax[1].set_xlim([interval[0], interval[1]])
        ax[2].set_xlim([interval[0], interval[1]])
        ax[0].grid(True, which='both')
        ax[1].grid(True, which='both')
        ax[2].grid(True, which='both')

        if log_scale:
            ax[0].set_xscale('log', base=10)
            ax[1].set_xscale('log', base=10)
            ax[2].set_xscale('log', base=10)

        plt.tight_layout()

        if test:
            save_plot(plt, 'prior', pdf=False, verbose=False)
        else:
            show_or_save_plot(plt, 'prior', transparent_background=True)
