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
from ..priors.prior import Prior
from ._posterior import Posterior
from ._gaussian_process_utilities import plot_training_convergence, \
    print_training_summary, plot_prediction, print_prediction_summary
from ..device._memory import Memory
from ..device._timer import Timer

__all__ = ['GaussianProcess']


# ================
# gaussian process
# ================

class GaussianProcess(object):
    """
    Gaussian process prior for regression.

    Parameters
    ----------

    mean : :class:`glearn.LinearModel`
        The mean function :math:`\\mu` of the regression.

    cov : :class:`glearn.Covariance`
        The covariance function :math:`\\Sigma` of the regression.

    See Also
    --------

    glearn.LinearModel
    glearn.Covariance

    Attributes
    ----------

    z : numpy.array
        An array of the size :math:`n` representing the training data.

    mean : :class:`glearn.LinearModel`
        An object representing the mean function of the Gaussian process prior.
        Once the model is trained, the hyperparameters of this object will be
        updated with their optimal values.

    cov : :class:`glearn.Covariance`
        An object representing the covariance function of the Gaussian process
        prior. Once the model is trained, the hyperparameters of this object
        will be updated with their optimal values.

    timer : :class:`glearn.Timer`
        A timer object that keeps track of process time and elapsed time of
        the training process.

    memory : :class:`glearn.Memory`
        A memory object that keeps track of resident memory acquired during the
        computation.

    training_result : dict
        A dictionary containing the training hyperparameters after the model
        is trained. This attribute is initially an empty dictionary, however,
        after calling :meth:`glearn.GaussianProcess.train`, this dictionary is
        updated and will have the following keys:

        * ``'config'``:
            * ``'max_bracket_trials'``: int, maximum number of trials to search
              for a bracket with sign-change in Chandrupatla root finding
              method. This option is relevant if ``optimization_method`` is
              set to ``chandrupatla``.
            * ``'max_iter'``: int, maximum number of iterations during
              optimization process.
            * ``'optimization_method'`` : {`'chandrupatla'`, `'brentq'`,
              `'Nelder-Mead'`, `'BFGS'`, `'CG'`, `'Newton-CG'`, `'dogleg'`,
              `'trust-exact'`, `'trust-ncg'`}, the optimization method.
            * ``'profile_hyperparam'`` : {`'none'`, `'var'`, `'var_noise'`},
              the profiling method of the likelihood function.
            * ``'tol'`` : float, the tolerance of convergence of
              hyperparameters during optimization.
            * ``'use_rel_error'``: bool, whether to use relative error or
              absolute error for the convergence criterion of the
              hyperparameters during the optimization.

        * ``'convergence'``:
            * ``'converged'``: numpy.array, an boolean array of the size of the
              number of the unknown hyperparameters, indicating which of the
              hyperparameters were converged according to the convergence
              criterion.
            * ``'errors'``: numpy.ndarray, a 2D array of the size of
              :math:`n_{\\mathrm{itr}} \\times, p` where
              :math:`n_{\\mathrm{itr}}` is the number of the optimization
              iterations and :math:`p` is the number of unknown
              hyperparameters. The :math:`i`-th row of this array is the error
              of the convergence of the hyperparameters at the :math:`i`-th
              iteration.
            * ``'hyperparams'``: numpy.ndarray, a 2D array of the size of
              :math:`n_{\\mathrm{itr}} \\times, p` where
              :math:`n_{\\mathrm{itr}}` is the number of the optimization
              iterations and :math:`p` is the number of unknown
              hyperparameters. The :math:`i`-th row of this array is the solved
              hyperparameters at the :math:`i`-th iteration.

        * ``'data'``:
            * ``'size'``: int, the size of the data.
            * ``'sparse'``: bool, determines whether the covariance matrix is
              sparse.
            * ``'nnz'``: int, the number of non-zero elements of the
              covariance matrix.
            * ``'avg_row_nnz'``: float, the average number of non-zero elements
              of the rows of the covariance matrix.
            * ``'density'``: float, the density of the covariance matrix, if
              sparse.
            * ``'dimension'``: int, the dimension of the space of data points.
            * ``'kernel_threshold'``: float, the threshold to tapper the
              kernel function to produce sparse covariance matrix.

        * ``'device'``:
            * ``'memory_usage'``: [int, str], a list where the first entry
              is the residence memory acquired during the computation, and the
              second entry of the list is the unit of the memory, which is
              one of ``'b'``, ``'Kb'``, ``'M'``, ``'Gb'``, etc.
            * ``'num_cpu_threads'``: int, number of CPU threads used during the
              computation.
            * ``'num_gpu_devices'``: int, number of GPU devices used during the
              computation.
            * ``'num_gpu_multiproc'``: int, number of GPU multi-processors per
              a GPU device.
            * ``'num_gpu_threads_per_multiproc'``: int, number of GPU threads
              per a GPU multiprocessors in each GPU device.

        * ``'hyperparam'``:
            * ``'eq_sigma'``: float, equivalent variance, which is
              :math:`\\sqrt{\\sigma^2 + \\varsigma^2}`.
            * ``'sigma'``: float, the hyperparameter :math:`\\sigma`.
            * ``'sigma0'``: float, the hyperparameter :math:`\\varsigma`.
            * ``'eta'``: float, the ratio
              :math:`\\eta = \\varsigma^2/\\sigma^2`.
            * ``'scale'``: numpy.array, the scale hyperparameters
              :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_d)`.

        * ``'optimization'``:
            * ``'max_fun'``: float, maximum value of the optimal likelihood
              function.
            * ``'message'``: str, message about optimization success or error.
            * ``'num_cor_eval'``: int, number of evaluations of the correlation
              matrix.
            * ``'num_fun_eval'``: int, number of evaluations of the likelihood
              function.
            * ``'num_jac_eval'``: int, number of evaluations of the Jacobian of
              the likelihood function.
            * ``'num_hes_eval'``: int, number of evaluations of the Hessian of
              the likelihood function.
            * ``'num_opt_eval'``: int, number of optimization iterations.

        * ``'time'``:
            * ``'cor_count'``: int, number of evaluations of correlation
              matrix.
            * ``'cor_proc_time'``: float, process time of evaluating
              correlation matrix.
            * ``'cor_wall_time'``: float, wall time of evaluating correlation
              matrix.
            * ``'det_count'``: int, number of evaluations of log-determinants.
            * ``'cor_proc_time'``: float, process time of evaluating
              log-determinants.
            * ``'det_wall_time'``: float, wall time of evaluating log
              determinants.
            * ``'lik_count'``: int, number of evaluations of the
              likelihood function.
            * ``'lik_proc_time'``: float, process time of evaluating the
              likelihood function.
            * ``'lik_wall_time'``: float, wall time of evaluating the
            * ``'opt_count'``: int, number of optimization iterations.
            * ``'opt_proc_time'``: float, process time of optimization.
            * ``'opt_wall_time'``: float, wall time of optimization.
            * ``'sol_count'``: int, number of solving linear systems.
            * ``'sol_proc_time'``: float, process time of linear systems.
            * ``'sol_wall_time'``: float, wall time of solving linear systems.
            * ``'trc_count'``: int, number of evaluating the trace of matrices.
            * ``'trc_proc_time'``: float, process time of evaluating the trace
              of matrices.
            * ``'trc_wall_time'``: float, wall time of evaluating the trace of
              matrices.

        * ``'imate_config'``:
            See ``info`` output of :func:`imate.logdet`, :func:`imate.trace`,
            or :func:`imate.traceinv`.

    prediction_result: dict
        A dictionary containing the prediction results. This attribute is
        initially empty, but once the function
        :meth:`glearn.GaussianProcess.predict` is called, this attribute is
        updated with the following keys:

        * ``'config'``:
            * ``'cov'``: bool, indicates whether the posterior covariance was
              computed.
            * ``'num_test_points'``: int, number of test points,
              :math:`n^{\\ast}`.
            * ``'num_training_points'``: int, number of training points
              :math:`n`.
        * ``'process'``:
            * ``'memory'``: [int, unit], a list that contains the used resident
              memory and its unit in the form of either of `b`, `Kb`, `Mb`,
              `Gb`, etc.
            * ``'proc_time'``: float, the process time of computing prediction.
            * ``'wall_time'``: float, the wall time of computing prediction.

    Methods
    -------

    train
    predict
    plot_likelihood

    Notes
    -----

    An instance of this class creates an object representing the Gaussian
    process :math:`\\mathcal{GP}(\\mu, \\Sigma)`, where :math:`\\mu` and
    :math:`\\Sigma` are respectively the mean and covariance functions of the
    Gaussian process model.

    **Training Algorithm:**

    The training method of this class is based on the algorithm described in
    [1]_.

    References
    ----------

    .. [1] Ameli, S., and Shadden. S. C. (2022). *Noise Estimation in Gaussian
       Process Regression*.
       `arXiv: 2206.09976 [cs.LG] <https://arxiv.org/abs/2206.09976>`_.

    Examples
    --------

    To define a Gaussian process object :math:`\\mathcal{GP}(\\mu, \\Sigma)`,
    first, an object for the linear model where :math:`\\mu` and an object
    for the covariance model :math:`\\Sigma` should be created as follows.

    **1. Generate Sample Training Data:**

    .. code-block:: python

        >>> import glearn
        >>> from glearn import sample_data

        >>> # Generate a set of training points
        >>> x = sample_data.generate_points(
        ...     num_points=30, dimension=1, grid=False,a=0.4, b=0.6,
        ...     contrast=0.9, seed=42)

        >>> # Generate noise sample data on the training points
        >>> y_noisy = glearn.sample_data.generate_data(x, noise_magnitude=0.1)

    **2. Create Linear Model:**

    Create an object for :math:`\\mu` function using
    :class:`glearn.LinearModel` class. On training points, the mean function is
    represented by the array

    .. math::

        \\boldsymbol{\\mu} = \\boldsymbol{\\phi}^{\\intercal}(\\boldsymbol{x})
        \\boldsymbol{\\beta}.

    .. code-block:: python

        >>> # Create mean object using glearn.
        >>> mean = glearn.LinearModel(x, polynomial_degree=2)

    **3. Create Covariance Object:**

    Create the covariance model using :class:`glearn.Covariance` class. On
    the training points, the covariance function is represented by the matrix

    .. math::

        \\boldsymbol{\\Sigma}(\\sigma, \\varsigma, \\boldsymbol{\\alpha}) =
        \\sigma^2 \\mathbf{K}(\\boldsymbol{\\alpha}) +
        \\varsigma^2 \\mathbf{I}.

    .. code-block:: python

        >>> # Define a Cauchy prior for scale hyperparameter
        >>> scale = glearn.priors.Cauchy()

        >>> # Create a covariance object
        >>> cov = glearn.Covariance(x, scale=scale)

    **4. Create Gaussian Process Object:**

    Putting all together, we can create an object for :math:`\\mathcal{GP}
    (\\mu, \\Sigma)` as follows:

    .. code-block:: python
        :emphasize-lines: 2

        >>> # Gaussian process object
        >>> gp = glearn.GaussianProcess(mean, cov)

    To train the model and predict on test data, see
    :meth:`glearn.GaussianProcess.train` and
    :meth:`glearn.GaussianProcess.predict`.

    Train the model to find the regression parameter
    :math:`\\boldsymbol{\\beta}` and the hyperparameters :math:`\\sigma`,
    :math:``
    """

    # ====
    # init
    # ====

    def __init__(self, mean, cov):
        """
        Constructor.
        """

        self.mean = mean
        self.cov = cov

        # Store member data
        self.z = None
        self.posterior = None
        self.training_result = None
        self.prediction_result = None
        self.w = None
        self.Y = None
        self.Mz = None

        # Counting elapsed wall time and cpu proc time
        self.timer = Timer()

        # Record resident memory (rss) of this current process in bytes
        self.memory = Memory()

    # ======================
    # check hyperparam guess
    # ======================

    def _check_hyperparam_guess(self, hyperparam_guess, profile_hyperparam):
        """
        Checks the input hyperparam, if not None.
        """

        # Find scale if not specifically given (as number, or array) the
        # training process will find scale as an unknown hyperparameter. But,
        # if scale is given, it leaves it out of hyperparameters.
        scale = self.cov.get_scale()

        # Number of parameters of covariance function
        if profile_hyperparam == 'none':
            # hyperparameters are sigma and sigma0
            num_cov_hyperparam = 2
        elif profile_hyperparam == 'var':
            # hyperparameter is eta
            num_cov_hyperparam = 1
        elif profile_hyperparam == 'var_noise':
            num_cov_hyperparam = 0
        else:
            raise ValueError('"profile_hyperparam" can be one of "none", ' +
                             '"var", or "var_noise".')

        # Convert hyperparam to numpy array
        if isinstance(hyperparam_guess, list):
            hyperparam_guess = numpy.array(hyperparam_guess)

        # Check number of hyperparameters
        if not isinstance(scale, (int, float, numpy.ndarray, list)):
            # Finds sigma, sigma0 (or eta), and all scale
            dimension = self.cov.mixed_cor.cor.points.shape[1]
            num_hyperparam = num_cov_hyperparam + dimension
        else:
            # Only find sigma and sigma0 (or eta)
            num_hyperparam = num_cov_hyperparam

        # check the size of input hyperparam_guess
        if hyperparam_guess.size != num_hyperparam:
            raise ValueError(
                'The size of "hyperparam_guess" (which is %d'
                % hyperparam_guess.size + ') does not match the number ' +
                'of hyperparameters (which is %d).' % num_hyperparam)

    # ==================
    # suggest hyperparam
    # ==================

    def _suggest_hyperparam(self, profile_hyperparam):
        """
        Suggests hyperparam_guess when it is None. ``hyperparam_guess`` may
        contain the following variables:

        * ``scale``: suggested from the mean, median, or peak of prior
          distributions for the scale hyperparam.
        * ``eta``: it uses the asymptotic relation that estimates eta before
          any computation is performed.
        * ``sigma`` and ``sigma0``: it assumes sigma is zero, and finds sigma0
          based on eta=infinity assumption.
        """

        # Find scale if not specifically given (as number, or array) the
        # training process will find scale as an unknown hyperparameter. But,
        # if scale is given, it leaves it out of hyperparameters.
        scale = self.cov.get_scale()

        # Set a default value for hyperparameter guess
        if isinstance(scale, (int, float, numpy.ndarray, list)):
            # Scale is given explicitly. No hyperparam is needed.
            scale_guess = []
        elif scale is None:

            # Get the prior of scale
            scale_prior = self.cov.cor.scale_prior

            if not isinstance(scale_prior, Prior):
                raise TypeError('"scale" should be given either explicitly ' +
                                'or as a prior distribution.')

            # Get the guess from the prior
            scale_guess = scale_prior.suggest_hyperparam(positive=True)

            if scale_guess <= 0.0:
                raise ValueError('The mean, median, or mode of the prior ' +
                                 'distribution for the scale' +
                                 'hyperparameter should be positive.')

            # Check type of scale guess
            if numpy.isscalar(scale_guess):
                scale_guess = numpy.array([scale_guess], dtype=float)
            elif isinstance(scale_guess, list):
                scale_guess = numpy.array(scale_guess, dtype=float)
            elif not isinstance(scale_guess, numpy.ndarray):
                raise TypeError('"scale_guess" should be a numpy array.')

            # Check if the size of scale guess matches the dimension
            dimension = self.cov.mixed_cor.cor.points.shape[1]
            if scale_guess.size != dimension:
                if scale_guess.size == 1:
                    scale_guess = numpy.tile(scale_guess, dimension)
                else:
                    raise ValueError('Size of "scale_guess" and "dimension" ' +
                                     'does not match.')

        # Other hyperparameters of covariance (except scale)
        if profile_hyperparam == 'none':
            # hyperparameters are sigma and sigma0. We assume all data is
            # noise, hence we set sigma to zero and solve sigma0 from
            # ordinary least square (OLS) solution.
            sigma_guess = 1e-2  # Small nonzero to avoid singularity
            sigma0_guess = self.posterior.likelihood.ols_solution()
            hyperparam_guess = numpy.r_[sigma_guess, sigma0_guess]

        elif profile_hyperparam == 'var':
            # Set scale before calling likelihood.asymptotic_maxima
            if len(scale_guess) > 0:
                self.posterior.likelihood.cov.set_scale(scale_guess)

            # hyperparameter is eta. Use asymptotic relations to guess eta
            asym_degree = 2
            asym_maxima = \
                self.posterior.likelihood.approx.maximaize_likelihood(
                        degree=asym_degree)

            if asym_maxima != []:
                eta_guess = asym_maxima[0]
            else:
                # In case no asymptotic root was found (all negative, complex)
                eta_guess = 1.0

            hyperparam_guess = numpy.r_[eta_guess]

        elif profile_hyperparam == 'var_noise':
            # No hyperparameter
            pass

        # Include scale guess
        if len(scale_guess) > 0:
            hyperparam_guess = numpy.r_[hyperparam_guess, scale_guess]

        return hyperparam_guess

    # =====
    # train
    # =====

    def train(
            self,
            z,
            hyperparam_guess=None,
            profile_hyperparam='var',
            log_hyperparam=True,
            optimization_method='Newton-CG',
            tol=1e-3,
            max_iter=1000,
            use_rel_error=True,
            imate_options={},
            gpu=False,
            verbose=False,
            plot=False):
        """
        Train the hyperparameters of the Gaussian process model.

        Parameters
        ----------

        z : numpy.array
            An array of the size :math:`n` representing the training data.

        hyperparam_guess : array_like or list, default=None
            A list (or array) of the initial guess for the hyperparameters of
            the Gaussian process model. The unknown hyperparameters depends on
            the following values for the argument ``profile_hyperparam``:

            * If ``profile_hyperparam=none``, then the hyperparameters are
              :math:`[\\sigma, \\varsigma]`,
              where :math:`\\sigma` and :math:`\\varsigma` are the standard
              deviations of the covariance model.
            * If ``profile_hyperparam=var``, then the hyperparameters are
              :math:`[\\eta]`, where :math:`\\eta = \\varsigma^2 / \\sigma^2`.
            * If ``profile_hyperparam=var_noise``, no guess is required.

            If no guess for either of the parameters are given (by setting
            ``hyperparam_guess=None``), an initial guess is generated using
            the asymptotic analysis algorithm described in [1]_.

        profile_hyperparam : {`'none'`, `'var'`, `'var_noise'`}, default:\
                `'var'`
            The type of likelihood profiling method to be used in optimization
            of the likelihood function.

            * ``'none'``: No profiling of the likelihood function. The
              optimization is performed in the full hyperparameter space.
              This is the standard method of optimizing the likelihood
              function.
            * ``'var'``: The variable variable :math:`\\sigma` is profiled
              in the likelihood function. This method is the fastest.
              The algorithm for this method can be found in [1]_.
            * ``'var_noise'``: Both variables :math:`\\sigma` and
              :math:`\\varsigma` are profiled in the likelihood function.
              The algorithm for this method can be found in [1]_.

        log_hyperparam : bool, default=True
            If `True`, the logarithm of the hyperparameters is used during the
            optimization process. This allows a greater search interval of the
            variables, making the optimization process more efficient.

        optimization_method : {`'chandrupatla'`, `'brentq'`, `'Nelder-Mead'`,\
                `'BFGS'`, `'CG'`, `'Newton-CG'`, `'dogleg'`, `'trust-exact'`,\
                `'trust-ncg'`}, default: `'Newton-CG'`
            The optimization method.

            * ``'chandrupatla'``: uses Jacobian only
            * ``'brentq'``: uses Jacobian only
            * ``'Nelder-Mead'``: uses function only (no derivative)
            * ``'BFGS'``: uses function and Jacobian
            * ``'CG'``: uses function and Jacobian
            * ``'Newton-CG'``: uses function, Jacobian, and Hessian
            * ``'dogleg'``: uses function, Jacobian, and Hessian
            * ``'trust-exact'``: uses function, Jacobian, and Hessian
            * ``'trust-ncg'``: uses function, Jacobian, and Hessian

            In the above methods, function, Jacobian, and Hessian refers to
            the likelihood function and its derivatives. The Jacobian and
            Hessian are computed automatically and the user does not need to
            provide them.

        tol : float, default: 1e-3
            The tolerance of convergence of hyperparameters during
            optimization. In case of multiple hyperparameters, the iterations
            stop once the convergence criterion is satisfied for all of the
            hyperparameters.

        max_iter : int, default: 1000
            Maximum number of iterations of the optimization process.

        use_rel_error : bool or None, default=True
            If `True`, the relative error is used for the convergence
            criterion. When `False`, the absolute error is used. When it is
            set to `None`, the callback function for minimize is not used.

        imate_options : dict, default={}
            A dictionary of options to pass arguments of the functions of the
            :mod:`imate` package, such as :func:`imate.logdet`,
            :func:`imate.trace`, and :func:`imate.traceinv`.

        gpu : bool, default=False
            If `True`, the computations are performed on GPU devices.
            Further setting on the GPU devices (such as the number of GPU
            devices) can be set by passing options to the :mod:`imate`
            package through ``imate_options`` argument.

        verbose : bool, default=False
            If `True`, verbose output on the optimization process is printer
            both during and after the computation.

        plot : bool, default=False
            If `True`, the likelihood or posterior function is plotted.

        See Also
        --------

        glearn.GaussianProcess.predict

        Notes
        -----

        **Maximum Posterior Method:**

        The training process maximizes the posterior function

        .. math::

            p(\\boldsymbol{\\beta}, \\sigma, \\varsigma,
            \\boldsymbol{\\alpha} \\vert z) =
            \\frac{ p(z \\vert \\boldsymbol{\\beta}, \\sigma, \\varsigma,
            \\boldsymbol{\\alpha})
            p(\\boldsymbol{\\beta}, \\sigma, \\varsigma, \\boldsymbol{\\alpha})
            }{p(z)}.

        The above hyperparameters are explained in the next section below.

        It is assumed that the hyperparameters are independent, namely

        .. math::

            p(\\boldsymbol{\\beta}, \\sigma, \\varsigma, \\boldsymbol{\\alpha})
            = p(\\boldsymbol{\\beta}) p(\\sigma) p(\\varsigma)
            p(\\boldsymbol{\\alpha})

        Also, it is assumed that :math:`p(\\sigma) = 1` and
        :math:`p(\\varsigma) = 1`.

        **Unknown Hyperparameters:**

        The unknown hyperparameters are as follows:

        * :math:`\\boldsymbol{\\beta}` from the linear model. A normal prior
          :math:`\\boldsymbol{\\beta} \\sim \\mathcal{N}(\\boldsymbol{b},
          \\mathbf{B})` for this hyperparameter can be set though
          :class:`glearn.LinearModel`. Once the model is trained, the optimal
          value of the posterior :math:`\\hat{\\boldsymbol{\\beta}} \\sim
          \\mathcal{N}(\\bar{\\boldsymbol{\\beta}},\\mathbf{C})` with the
          posterior mean and posterior covariance of this hyperparameter can be
          obtained by ``GaussianProcess.mean.beta`` and
          ``GaussianProcess.mean.C`` attributes.

        * :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\dots, \\alpha_d)`,
          where :math:`d` is the dimension of the space. A prior
          :math:`p(\\boldsymbol{\\alpha})` or an initial guess for this
          hyperparameter can be set by the argument ``scale`` in
          :class:`glearn.Covariance` class. The posterior value
          :math:`\\hat{\\boldsymbol{\\alpha}}` of this hyperparameter can be
          accessed by :meth:`glearn.Covariance.get_scale` function on the
          attribute ``GaussianProcess.cov`` covariance object.

        * :math:`\\sigma` and :math:`\\varsigma`, which an initial guess for
          these hyperparameters can be set by ``hyperparam_guess`` argument to
          the :meth:`glearn.GaussianProcess.train` function. The optimal
          estimated values :math:`\\hat{\\sigma}` and :math:`\\hat{\\varsigma}`
          of these hyperparameters can be found by the
          :meth:`glearn.Covariance.get_sigmas` function on the
          ``GaussianProcess.cov`` attribute.

        **Profile Likelihood:**

        The profile likelihood reduces the dimension of the space of the
        unknown hyperparameters.

        * When ``profile_likelihood`` is set to ``none``, the likelihood
          function explicitly depends on the two hyperparameters
          :math:`\\sigma` and :math:`\\varsigma`.
        * When ``profile_likelihood`` is set to ``var``, the likelihood
          function depends on the two hyperparameters
          :math:`\\eta=\\varsigma^2/\\sigma^2`, which is profiles over
          the hyperparameter :math:`\\sigma`, reducing the number of the
          hyperparameters by one.
        * When ``profile_likelihood`` is set to ``var_sigma``, the
          likelihood function is profiles over both :math:`\\sigma` and
          :math:`\\eta`, reducing the number of unknown hyperparameters
          by two.

        References
        ----------

        .. [1] Ameli, S., and Shadden. S. C. (2022). *Noise Estimation in
               Gaussian Process Regression*.
               `arXiv: 2206.09976 [cs.LG] <https://arxiv.org/abs/2206.09976>`_.

        Examples
        --------

        To define a Gaussian process object :math:`\\mathcal{GP}(\\mu,
        \\Sigma)`, first, an object for the linear model where :math:`\\mu` and
        an object for the covariance model :math:`\\Sigma` should be created as
        follows.

        **1. Generate Sample Training Data:**

        .. code-block:: python

            >>> import glearn
            >>> from glearn import sample_data

            >>> # Generate a set of training points
            >>> x = sample_data.generate_points(
            ...     num_points=30, dimension=1, grid=False,a=0.4, b=0.6,
            ...     contrast=0.9, seed=42)

            >>> # Generate noise sample data on the training points
            >>> y_noisy = glearn.sample_data.generate_data(
            ...     x, noise_magnitude=0.1)

        **2. Create Linear Model:**

        Create an object for :math:`\\mu` function using
        :class:`glearn.LinearModel` class. On training points, the mean
        function is represented by the array

        .. math::

            \\boldsymbol{\\mu} = \\boldsymbol{\\phi}^{\\intercal}
            (\\boldsymbol{x}) \\boldsymbol{\\beta}.

        .. code-block:: python

            >>> # Create mean object using glearn.
            >>> mean = glearn.LinearModel(x, polynomial_degree=2)

        **3. Create Covariance Object:**

        Create the covariance model using :class:`glearn.Covariance` class. On
        the training points, the covariance function is represented by the
        matrix

        .. math::

            \\boldsymbol{\\Sigma}(\\sigma, \\varsigma, \\boldsymbol{\\alpha}) =
            \\sigma^2 \\mathbf{K}(\\boldsymbol{\\alpha}) +
            \\varsigma^2 \\mathbf{I}.

        .. code-block:: python

            >>> # Define a Cauchy prior for scale hyperparameter
            >>> scale = glearn.priors.Cauchy()

            >>> # Create a covariance object
            >>> cov = glearn.Covariance(x, scale=scale)

        **4. Create Gaussian Process Object:**

        Putting all together, we can create an object for :math:`\\mathcal{GP}
        (\\mu, \\Sigma)` as follows:

        .. code-block:: python

            >>> # Gaussian process object
            >>> gp = glearn.GaussianProcess(mean, cov)

        **5. Train The Model:**

        Train the model to find the regression parameter
        :math:`\\boldsymbol{\\beta}` and the hyperparameters :math:`\\sigma`,
        :math:`\\varsigma`, and :math:`\\boldsymbol{\\alpha}`.

        .. code-block:: python
            :emphasize-lines: 2, 3, 4, 5, 6

            >>> # Train
            >>> result = gp.train(
            ...     y_noisy, profile_hyperparam='var', log_hyperparam=True,
            ...     hyperparam_guess=None, optimization_method='Newton-CG',
            ...     tol=1e-2, max_iter=1000, use_rel_error=True,
            ...     imate_options={'method': 'cholesky'}, verbose=True)

        The results of training process can be found in the ``training_result``
        attribute as follows:

        .. code-block:: python

            >>> # Training results
            >>> gp.training_results
            {
                'config': {
                    'max_bracket_trials': 6,
                    'max_iter': 1000,
                    'optimization_method': 'Newton-CG',
                    'profile_hyperparam': 'var',
                    'tol': 0.001,
                    'use_rel_error': True
                },
                'convergence': {
                    'converged': array([ True,  True]),
                    'errors': array([[       inf,        inf],
                                     [0.71584751, 0.71404119],
                                     ...
                                     [0.09390544, 0.07001806],
                                     [0.        , 0.        ]]),
                    'hyperparams': array([[-0.39474532,  0.39496465],
                                         [-0.11216787,  0.67698568],
                                         ...
                                         [ 2.52949461,  3.31844015],
                                         [ 2.52949461,  3.31844015]])
                },
                'data': {
                    'avg_row_nnz': 30.0,
                    'density': 1.0,
                    'dimension': 1,
                    'kernel_threshold': None,
                    'nnz': 900,
                    'size': 30,
                    'sparse': False
                },
                'device': {},
                'hyperparam': {
                    'eq_sigma': 0.0509670753735289,
                    'eta': 338.4500691178316,
                    'scale': array([2081.80547198]),
                    'sigma': 0.002766315834132389,
                    'sigma0': 0.05089194699396757
                },
                'imate_config': {
                    'device': {
                        'num_cpu_threads': 8,
                        'num_gpu_devices': 0,
                        'num_gpu_multiprocessors': 0,
                        'num_gpu_threads_per_multiprocessor': 0
                    },
                    'imate_interpolate': False,
                    'imate_method': 'cholesky',
                    'imate_tol': 1e-08,
                    'max_num_samples': 0,
                    'min_num_samples': 0,
                    'solver': {
                        'cholmod_used': False,
                        'method': 'cholesky',
                        'version': '0.18.2'
                    }
                },
                'optimization': {
                    'max_fun': 42.062003754316756,
                    'message': 'Optimization terminated successfully.',
                    'num_cor_eval': 45,
                    'num_fun_eval': 15,
                    'num_hes_eval': 15,
                    'num_jac_eval': 15,
                    'num_opt_iter': 15
                },
                'time': {
                    'cor_count': 45,
                    'cor_proc_time': 5.494710099000002,
                    'cor_wall_time': 0.7112228870391846,
                    'det_count': 15,
                    'det_proc_time': 0.013328177999998747,
                    'det_wall_time': 0.002137899398803711,
                    'lik_count': 45,
                    'lik_proc_time': 6.337607392999995,
                    'lik_wall_time': 0.8268890380859375,
                    'opt_proc_time': 6.542538159,
                    'opt_wall_time': 0.8506159782409668,
                    'sol_count': 105,
                    'sol_proc_time': 1.9085943260000005,
                    'sol_wall_time': 0.24165892601013184,
                    'trc_count': 30,
                    'trc_proc_time': 0.05656320299999962,
                    'trc_wall_time': 0.006264448165893555
                }
            }

        **Verbose Output:**

        By setting ``verbose`` to `True`, useful info about the process is
        printed.

        .. literalinclude:: ../_static/data/glearn.gp.train-verbose.txt
            :language: python

        **Obtaining the Optimal Hyperparameter:**

        Once the model is trained, the optimal regression parameter
        :math:`\\boldsymbol{\\beta}` of the mean function, the variance
        hyperparameters :math:`\\sigma` and :math:`\\varsigma` of the
        covariance, and the scale hyperparameters :math:`\\boldsymbol{\\alpha}`
        of the covariance can be accessed as follows.

        .. code-block:: python

            >>> # Getting beta
            >>> gp.mean.beta
            [ 0.07843029  3.75650903 -3.68907446]

            >>> # Getting variances sigma and varsigma
            >>> gp.cov.get_sigmas()
            (8.751267455041524e-05, 0.11059589121331345)

            >>> # Getting the scale parameter alpha
            gp.cov.get_scale()
            [0.00032829]

        **Plotting:**

        Plotting the convergence of the hyperparameters:

        .. code-block:: python
            :emphasize-lines: 7

            >>> # Train
            >>> result = gp.train(
            ...     y_noisy, profile_hyperparam='var', log_hyperparam=True,
            ...     hyperparam_guess=None, optimization_method='Newton-CG',
            ...     tol=1e-6, max_iter=1000, use_rel_error=True,
            ...     imate_options={'method': 'cholesky'}, verbose=True,
            ...     plot=True)

        .. image:: ../_static/images/plots/gp_convergence.png
            :align: center
            :width: 75%
            :class: custom-dark

        Note that since we set ``log_hyperparam=True``, the logarithm of the
        scale hyperparameter, :math:`\\log \\alpha_1`, is used in the
        optimization process, as can be seen in the legend of the figure. Also,
        the iterations stop once the convergence error reaches the specified
        tolerance ``tol=1e-2``.

        **Passing Initial Guess for Hyperparameters:**

        One can set an initial guess for hyperparameters by passing the
        argument ``hyperparam_guess``. Since in the above example, the
        argument ``profile_hyperparam`` is set to ``var``, the unknown
        hyperparameters are

        .. math::

            (\\eta, \\alpha_1, \\dots, \\alpha_d),

        where :math:`d` is the dimension of the space, here :math:`d=1`.
        Suppose we guess :math:`\\sigma=0.1`, :math:`\\varsigma=1`, which makes
        :math:`\\eta = \\varsigma^2/\\sigma^2 = 100`. We also set an initial
        guess :math:`\\alpha_1 = 1` for the scale hyperparameter.

        .. code-block:: python
            :emphasize-lines: 5

            >>> # Train
            >>> hyperparam_guess = [100, 0.5]
            >>> result = gp.train(
            ...     y_noisy, profile_hyperparam='var',
            ...     hyperparam_guess=hyperparam_guess)

            >>> # Getting variances sigma and varsigma
            >>> gp.cov.get_sigmas()
            (8.053463077699365e-05, 0.11059589693864141)

        **Using Other Profile Likelihood Methods:**

        Here we use no profiling method (``profile_likelihood=none``) and we
        pass the hyperparameter guesses :math:`(\\sigma, \\varsigma) = (0.1,
        1, 0.5)`.


        .. code-block:: python
            :emphasize-lines: 4, 5

            >>> # Train
            >>> hyperparam_guess = [0.1, 1, 0.5]
            >>> result = gp.train(
            ...     y_noisy, profile_hyperparam='none',
            ...     hyperparam_guess=hyperparam_guess)

            >>> # Getting variances sigma and varsigma
            >>> gp.cov.get_sigmas()
            (0.0956820228647455, 0.11062417694050758)
        """

        # Set self.cov with imate options
        self.cov.set_imate_options(imate_options)

        # Create a posterior object. Note that self.posterior should be defined
        # before calling self._check_hyperparam_guess
        self.posterior = Posterior(self.mean, self.cov, z,
                                   profile_hyperparam=profile_hyperparam,
                                   log_hyperparam=log_hyperparam)

        # Reset function evaluation counters and timers
        self.posterior.reset()

        # Prepare or suggest hyperparameter guess
        if hyperparam_guess is not None:
            self._check_hyperparam_guess(hyperparam_guess, profile_hyperparam)

            # Convert hyperparam_guess to numpy array
            if isinstance(hyperparam_guess, list):
                hyperparam_guess = numpy.array(hyperparam_guess, dtype=float)
            elif numpy.isscalar(hyperparam_guess):
                hyperparam_guess = numpy.array([hyperparam_guess])

        else:
            hyperparam_guess = self._suggest_hyperparam(
                    profile_hyperparam)

        # Maximize posterior w.r.t hyperparameters
        self.training_result = self.posterior.maximize_posterior(
                hyperparam_guess, optimization_method=optimization_method,
                tol=tol, max_iter=max_iter, use_rel_error=use_rel_error,
                verbose=verbose)

        if plot:
            plot_training_convergence(
                    self.posterior, self.training_result, verbose)

        if verbose:
            print_training_summary(self.training_result)

        # Set optimal parameters (sigma, sigma0) to covariance object
        sigma = self.training_result['hyperparam']['sigma']
        sigma0 = self.training_result['hyperparam']['sigma0']
        self.cov.set_sigmas(sigma, sigma0)

        # Set optimal parameters (b and B) to mean object
        self.mean.update_hyperparam(self.cov, z)

        # Store data for future reference
        self.z = z

        return self.training_result

    # ===============
    # plot likelihood
    # ===============

    def plot_likelihood(
            self,
            z=None,
            profile_hyperparam='var'):
        """
        Plot the likelihood function and its derivatives with respect to the
        hyperparameters.

        Parameters
        ----------

        z : numpy.array
            An array of the size :math:`n` representing the training data.
            If `z` is not provided (set to `None`), the training data that used
            to define the :class:`glearn.GaussianProcess` class is used.

        profile_hyperparam : {`'none'`, `'var'`, `'var_noise'`}, default:\
                `'var'`
            The type of likelihood profiling method to be used in optimization
            of the likelihood function.

            * When ``profile_likelihood`` is set to ``none``, the likelihood
              function explicitly depends on the two hyperparameters
              :math:`\\sigma` and :math:`\\varsigma`.
            * When ``profile_likelihood`` is set to ``var``, the likelihood
              function depends on the two hyperparameters
              :math:`\\eta=\\varsigma^2/\\sigma^2`, which is profiles over
              the hyperparameter :math:`\\sigma`, reducing the number of the
              hyperparameters by one.
            * When ``profile_likelihood`` is set to ``var_sigma``, the
              likelihood function is profiles over both :math:`\\sigma` and
              :math:`\\eta`, reducing the number of unknown hyperparameters
              by two.

        Notes
        -----

        This function plots likelihood function and its first and second
        derivatives with respect to the hyperparameters. The type of
        hyperparameters depend on the profiling method which is set by
        ``profile_hyperparam``.

        This function is only used for testing purposes, and can only plot
        1D and 2D data.

        .. warning::

            This function may take a long time, and is only used for testing
            purposes on small datasets.

        Note that the maximum points of the likelihood plots may not
        correspond to the optimal values of the hyperparameters. This is
        because the hyperparameters are found by the maximum points of the
        posterior function. If the prior for the scale hyperparameter is the
        uniform distribution, then the likelihood and the posterior functions
        are them same and the maxima of the likelihood function in the plots
        correspond to the optimal hyperparameters.

        Examples
        --------

        To define a Gaussian process object :math:`\\mathcal{GP}(\\mu,
        \\Sigma)`, first, an object for the linear model where :math:`\\mu` and
        an object for the covariance model :math:`\\Sigma` should be created as
        follows.

        **1. Generate Sample Training Data:**

        .. code-block:: python

            >>> import glearn
            >>> from glearn import sample_data

            >>> # Generate a set of training points
            >>> x = sample_data.generate_points(
            ...     num_points=30, dimension=1, grid=False,a=0.4, b=0.6,
            ...     contrast=0.9, seed=42)

            >>> # Generate noise sample data on the training points
            >>> y_noisy = glearn.sample_data.generate_data(
            ...     x, noise_magnitude=0.1)

        **2. Create Linear Model:**

        Create an object for :math:`\\mu` function using
        :class:`glearn.LinearModel` class. On training points, the mean
        function is represented by the array

        .. math::

            \\boldsymbol{\\mu} = \\boldsymbol{\\phi}^{\\intercal}
            (\\boldsymbol{x}) \\boldsymbol{\\beta}.

        .. code-block:: python

            >>> # Create mean object using glearn.
            >>> mean = glearn.LinearModel(x, polynomial_degree=2)

        **3. Create Covariance Object:**

        Create the covariance model using :class:`glearn.Covariance` class. On
        the training points, the covariance function is represented by the
        matrix

        .. math::

            \\boldsymbol{\\Sigma}(\\sigma, \\varsigma, \\boldsymbol{\\alpha}) =
            \\sigma^2 \\mathbf{K}(\\boldsymbol{\\alpha}) +
            \\varsigma^2 \\mathbf{I}.

        .. code-block:: python

            >>> # Define a Cauchy prior for scale hyperparameter
            >>> scale = glearn.priors.Cauchy()

            >>> # Create a covariance object
            >>> cov = glearn.Covariance(x, scale=scale)

        **4. Create Gaussian Process Object:**

        Putting all together, we can create an object for :math:`\\mathcal{GP}
        (\\mu, \\Sigma)` as follows:

        .. code-block:: python

            >>> # Gaussian process object
            >>> gp = glearn.GaussianProcess(mean, cov)

        **5. Train The Model:**

        Train the model to find the regression parameter
        :math:`\\boldsymbol{\\beta}` and the hyperparameters :math:`\\sigma`,
        :math:`\\varsigma`, and :math:`\\boldsymbol{\\alpha}`.

        .. code-block:: python

            >>> # Train
            >>> result = gp.train(
            ...     y_noisy, profile_hyperparam='var', log_hyperparam=True,
            ...     hyperparam_guess=None, optimization_method='Newton-CG',
            ...     tol=1e-2, max_iter=1000, use_rel_error=True,
            ...     imate_options={'method': 'cholesky'}, verbose=True)

        **Plotting:**

        After the  training when the hyperparameters were tuned, plot the
        likelihood function as follows:

        .. code-block:: python

            >>> # Plot likelihood function and its derivatives
            >>> gp.plot_likelihood()

        .. image:: ../_static/images/plots/gp_likelihood_var_1.png
            :align: center
            :width: 100%
            :class: custom-dark

        .. image:: ../_static/images/plots/gp_likelihood_var_2.png
            :align: center
            :width: 100%
            :class: custom-dark

        .. image:: ../_static/images/plots/gp_likelihood_var_3.png
            :align: center
            :width: 100%
            :class: custom-dark

        If we set ``profile_likelihood=none``, the followings will be plotted
        instead:

        .. image:: ../_static/images/plots/gp_likelihood_none_1.png
            :align: center
            :width: 100%
            :class: custom-dark

        .. image:: ../_static/images/plots/gp_likelihood_none_2.png
            :align: center
            :width: 100%
            :class: custom-dark

        .. image:: ../_static/images/plots/gp_likelihood_none_3.png
            :align: center
            :width: 100%
            :class: custom-dark
        """

        if z is None:
            if self.z is None:
                raise ValueError('Data "z" cannot be None.')
            z = self.z

        if (self.training_result is None) or \
           (self.training_result['config']['profile_hyperparam'] !=
                profile_hyperparam):

            # Train
            self.training_result = self.train(
                z, hyperparam_guess=None,
                profile_hyperparam=profile_hyperparam, log_hyperparam=True,
                optimization_method='Newton-CG', tol=1e-3, use_rel_error=True,
                verbose=False, plot=False)

            # Create a posterior object
            self.posterior = Posterior(self.mean, self.cov, z,
                                       profile_hyperparam=profile_hyperparam,
                                       log_hyperparam=True)

        # Plot likelihood
        self.posterior.likelihood.plot(self.training_result)

    # =======
    # predict
    # =======

    def predict(
            self,
            test_points,
            cov=False,
            plot=False,
            true_data=None,
            confidence_level=0.95,
            verbose=False):
        """
        Make prediction on test points.

        .. note::

            This function should be called after training the model with
            :meth:`glearn.GaussianProcess.train`.

        Parameters
        ----------

        test_points : numpy.array
            An array of the size :math:`n^{\\ast} \\times d` representing the
            coordinates of :math:`n^{\\ast}` training points in the :math:`d`
            dimensional space. Each row of this array is the coordinate of a
            test point.

        cov : bool, default=False
            If `True`, the prediction includes both the posterior mean and
            posterior covariance. If `False`, only the posterior mean is
            computed. Note that obtaining posterior covariance is
            computationally expensive, so if it is not needed, set this
            argument to `False`.

        plot : bool, default=False
            If `True`, plots the prediction results.

        true_data : numpy.array, default=None
            An array of the size :math:`n^{\\ast}` of the true values of the
            test points data (if known). This option is used for benchmarking
            purposes only to compare the prediction results with their true
            values if known. If ``plot`` is set to `True`, the true values
            together with the error of the prediction with respect to the
            true data is also plotted.

        confidence_level : float, default=0.95
            The confidence level that determines the confidence interval of the
            error of the prediction.

        verbose : bool, default=False
            If `True`, it prints information about the result.

        See Also
        --------

        glearn.GaussianProcess.train

        Notes
        -----

        Consider the Gaussian process prior on the function :math:`y(x)` by

        .. math::

            y \\sim \\mathcal{GP}(\\mu, \\Sigma).

        This means that on the training points :math:`\\boldsymbol{x}`, the
        array of training data :math:`\\boldsymbol{y}` has the normal
        distribution

        .. math::

            \\boldsymbol{y} \\sim \\mathcal{N}(\\boldsymbol{\\mu},
            \\boldsymbol{\\Sigma}),

        where :math:`\\boldsymbol{\\mu}` is the array of the mean and
        :math:`\\boldsymbol{\\Sigma}` is the covariance matrix of the training
        points.

        On some test points :math:`\\boldsymbol{x}^{\\ast}`, the posterior
        predictive distribution of the prediction
        :math:`\\boldsymbol{y}^{\\ast}(\\boldsymbol{x}^{\\ast})` has the
        distribution

        .. math::

            \\boldsymbol{y}^{\\ast}(\\boldsymbol{x}^{\\ast}) \\sim \\mathcal{N}
            \\left(\\boldsymbol{\\mu}^{\\ast}(\\boldsymbol{x}^{\\ast}),
            \\mathbf{\\Sigma}^{\\ast \\ast} (\\boldsymbol{x}^{\\ast},
            \\boldsymbol{x}'^{\\ast})\\right.

        where:

        * :math:`\\boldsymbol{\\mu}^{\\ast}` is the posterior predictive mean.
        * :math:`\\mathbf{\\Sigma}^{\\ast \\ast}` is the posterior predictive
          covariance between test points and themselves.

        This function finds :math:`\\boldsymbol{\\mu}^{\\ast}` and
        :math:`\\boldsymbol{\\Sigma}^{\\ast \\ast}`.

        **Computational Complexity:**

        The first call to :meth:`glearn.GaussianProcess.predict` has the
        computational complexity of :math:`\\mathcal{O}((n^3 + n^{\\ast})`,
        where :math:`n` is the number of training points and :math:`n^{\\ast}`
        is the number of test points.

        After the first call to this function, all coefficients during the
        computations that are independent of the test points are stored in
        the Gaussian process object, which can be reused for the future calls.
        The computational complexity of the future calls to this function is
        as follows.

        * If posterior covariance is computed (by setting ``cov=True``), the
          computational complexity is still the same as the first call to the
          prediction function, namely, :math:`\\mathcal{O}(n^3 + n^{\\ast})`.
        * If posterior covariance is not computed (by setting ``cov=False``),
          the computational complexity is :math:`\\mathcal{O}(n^{\\ast})`.
          See this point in the example below.

        Examples
        --------

        To define a Gaussian process object :math:`\\mathcal{GP}(\\mu,
        \\Sigma)`, first, an object for the linear model where :math:`\\mu` and
        an object for the covariance model :math:`\\Sigma` should be created as
        follows.

        **1. Generate Sample Training Data:**

        .. code-block:: python

            >>> import glearn
            >>> from glearn import sample_data

            >>> # Generate a set of training points
            >>> x = sample_data.generate_points(
            ...     num_points=30, dimension=1, grid=False,a=0.4, b=0.6,
            ...     contrast=0.9, seed=42)

            >>> # Generate noise sample data on the training points
            >>> y_noisy = glearn.sample_data.generate_data(
            ...     x, noise_magnitude=0.1)

        **2. Create Linear Model:**

        Create an object for :math:`\\mu` function using
        :class:`glearn.LinearModel` class. On training points, the mean
        function is represented by the array

        .. math::

            \\boldsymbol{\\mu} = \\boldsymbol{\\phi}^{\\intercal}
            (\\boldsymbol{x}) \\boldsymbol{\\beta}.

        .. code-block:: python

            >>> # Create mean object using glearn.
            >>> mean = glearn.LinearModel(x, polynomial_degree=2)

        **3. Create Covariance Object:**

        Create the covariance model using :class:`glearn.Covariance` class. On
        the training points, the covariance function is represented by the
        matrix

        .. math::

            \\boldsymbol{\\Sigma}(\\sigma, \\varsigma, \\boldsymbol{\\alpha}) =
            \\sigma^2 \\mathbf{K}(\\boldsymbol{\\alpha}) +
            \\varsigma^2 \\mathbf{I}.

        .. code-block:: python

            >>> # Define a Cauchy prior for scale hyperparameter
            >>> scale = glearn.priors.Cauchy()

            >>> # Create a covariance object
            >>> cov = glearn.Covariance(x, scale=scale)

        **4. Create Gaussian Process Object:**

        Putting all together, we can create an object for :math:`\\mathcal{GP}
        (\\mu, \\Sigma)` as follows:

        .. code-block:: python

            >>> # Gaussian process object
            >>> gp = glearn.GaussianProcess(mean, cov)

        **5. Train The Model:**

        Train the model to find the regression parameter
        :math:`\\boldsymbol{\\beta}` and the hyperparameters :math:`\\sigma`,
        :math:`\\varsigma`, and :math:`\\boldsymbol{\\alpha}`.

        .. code-block:: python

            >>> # Train
            >>> result = gp.train(y_noisy)

        **6. Predict on Test Points:**

        After training the hyperparameters, the ``gp`` object is ready to
        predict the data on new points. First, we create a set of :math:`1000`
        test points :math:`\\boldsymbol{x}^{\\ast}` equally distanced in the
        interval :math:`[0, 1]`.

        .. code-block:: python

            >>> # Generate test points
            >>> test_points = sample_data.generate_points(num_points=1000,
            ...     dimension=1, grid=True)

        For the purpose of comparison, we also generate the noise-free data on
        the test points, :math:`y(\\boldsymbol{x}^{\\ast})`, using zero noise
        :math:`\\varsigma = 0`.

        .. code-block:: python

            >>> # True data (without noise)
            >>> y_true = sample_data.generate_data(test_points,
            ...     noise_magnitude=0.0)

        Note that the above step is unnecessary and only used for the purpose
        of comparison with the prediction since we already know the exact
        function that generated the noisy data :math:`y` in the first place.

        .. code-block:: python
            :emphasize-lines: 2, 3, 4

            >>> # Prediction on test points
            >>> y_star_mean, y_star_cov = gp.predict(
            ...     test_points, cov=True, plot=True, confidence_level=0.95,
            ...     true_data=y_true, verbose=True)

        Some information about the prediction process can be found by
        ``prediction_result`` attribute of ``GaussianProcess`` class as
        follows:

        .. code-block:: python

            >>> # Prediction result
            >>> gp.prediction_result
            {
                'config': {
                    'cov': True,
                    'num_test_points': 1000,
                    'num_training_points': 30
                },
                'process': {
                    'memory': [24768512, 'b'],
                    'proc_time': 0.9903236419999999,
                    'wall_time': 0.13731884956359863
                }
            }

        **Verbose Output:**

        By setting ``verbose`` to `True`, useful info about the process is
        printed.

        .. literalinclude:: ../_static/data/glearn.gp.predict-verbose.txt
            :language: python

        **Plotting:**

        By setting ``plot`` to `True`, the prediction is plotted as follows.

        .. code-block:: python
            :emphasize-lines: 4

            >>> # Prediction on test points
            >>> y_star_mean, y_star_cov = gp.predict(
            ...     test_points, cov=True, plot=True, confidence_level=0.95,
            ...     true_data=y_true, Plot=True)

        .. image:: ../_static/images/plots/gp_predict.png
            :align: center
            :width: 75%
            :class: custom-dark

        **Further Call to Prediction Function:**

        One of the features of this function is that, once a prediction on a
        set of test points is made, a second and further call to this
        prediction function consumes significantly less processing time,
        provided that we only compute the posterior mean (and not the
        posterior covariance). This result holds even if the future alls are on
        a different set of test points.

        To see this, let print the process time on the previous prediction:

        .. code-block:: python

            >>> # Process time and time
            >>> gp.prediction_result['process']['proc_time']
            0.9903236419999999

        Now, we made prediction on a different set of test points generated
        randomly and measure the process time:

        .. code-block:: python

            >>> # Generate test points
            >>> test_points_2 = sample_data.generate_points(num_points=1000,
            ...     dimension=1, grid=False)

            >>> # Predict posterior mean on the new set of test points
            >>> y_star_mean_2 = gp.predict(test_points_2, cov=False)

            >>> # Measure the process time
            >>> gp.prediction_result['process']['proc_time']
            0.051906865999999496

        The above process time is significantly less than the first call to
        the prediction function. This is because the first call to the
        prediction function computes the prediction coefficients that are
        independent of the test points. A future call to the prediction
        function reuses the coefficients and makes the prediction at
        :math:`\\mathcal{O}(n^{\\ast})` time.
        """

        if self.z is None:
            raise RuntimeError('Data should be trained first before calling ' +
                               'the predict function.')

        # If test points are 1d array, wrap them to a 2d array
        if test_points.ndim == 1:
            test_points = numpy.array([test_points], dtype=float).T

        if test_points.shape[1] != self.mean.points.shape[1]:
            raise ValueError('"test_points" should have the same dimension ' +
                             'as the training points.')

        # Record the used memory of the current process at this point in bytes
        self.timer.reset()
        self.timer.tic()
        self.memory.reset()
        self.memory.start()

        # Design matrix on test points
        X_star = self.mean.generate_design_matrix(test_points)

        # Covariance on data points to test points
        cov_star = self.cov.cross_covariance(test_points)

        beta = self.mean.beta
        X = self.mean.X

        # w, Y, and Mz are computed once per data z and are independent of the
        # test points. On the future calls for the prediction on test points,
        # these will not be computed again.
        if (self.w is None) or (self.Y is None) or (self.Mz is None):

            # Solve Sinv * z and Sinv * X
            self.w = self.cov.solve(self.z)
            self.Y = self.cov.solve(X)

            # Compute Mz (Note: if b is zero, the following is actually Mz, but
            # if b is not zero, the following is Mz + C*Binv*b)
            self.Mz = self.w - numpy.matmul(self.Y, beta)

        # Posterior predictive mean. Note that the following uses the dual
        # formulation, that is, z_star at test point is just the dot product
        # of qualities (w, Mz) that are independent of the test point and they
        # were computed once.
        z_star_mean = cov_star.T.dot(self.Mz) + X_star.dot(beta)

        # Compute posterior predictive covariance
        z_star_cov = None
        if cov:

            # Compute R
            R = X_star.T - self.Y.T @ cov_star

            # Covariance on test points to test points
            cov_star_star = self.cov.auto_covariance(test_points)

            # Posterior covariance of beta
            C = self.mean.C

            if C is None:
                raise RuntimeError('Parameters of LinearModel are None. ' +
                                   'Call "train" function first.')

            # Covariance of data points to themselves
            Sinv_cov_star = self.cov.solve(cov_star)

            # Posterior predictive covariance
            z_star_cov = cov_star_star - cov_star.T @ Sinv_cov_star + \
                numpy.matmul(R.T, numpy.matmul(C, R))

        # Stop timer
        self.timer.toc()
        self.memory.stop()

        # Read memory and its unit
        mem_diff, mem_unit = self.memory.get_mem()

        self.prediction_result = {
            'config': {
                'num_training_points': self.z.size,
                'num_test_points': test_points.shape[0],
                'cov': cov
            },
            'process': {
                'wall_time': self.timer.wall_time,
                'proc_time': self.timer.proc_time,
                'memory': [mem_diff, mem_unit]
            }
        }

        # Print summary
        if verbose:
            print_prediction_summary(self.prediction_result)

        # Plot prediction
        if plot:
            plot_prediction(self.mean.points, test_points, self.z, z_star_mean,
                            z_star_cov, confidence_level, true_data, verbose)

        if cov:
            return z_star_mean, z_star_cov
        else:
            return z_star_mean
