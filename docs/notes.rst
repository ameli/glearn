*****
Notes
*****

* In profile likelihood, the last term of `lp` is a constant. `zMz`, at the
  optimal value of sigma**2, is constant. This is only for profile likelihood
  where optimal value of sigma is used.

====
ToDo
====

----------------
hyperparam guess
----------------

* For the guess of eta hyperparam, use the asymptotic relation.
* For sigma and sigma0 hypepraram, assume sigma is zero, and use eta=inf to
  find optimal sigma0 as a hyperparam.

--------------------
Asymptotic Relations
--------------------

* in ``_profile_likelihood.py`` in ``_compute_assymptotre_der1_eta``, the
  matrix ``R - I-Q`` is dense. Instead of computing it directly, perform the
  dot product of ``R`` w.r.t a vector. For trace of ``N``, use cyclic property
  of trace.
* Use asymptotic derivatives w.r.t eta to find ``eta_guess`` (used in
  ``hyperparam_guess``) when user does not provide a guess for eta.

  How to modify implementation:

  Create a function ``asymptote_polynomial`` to find the coefficients
  [a0, a1, a2, a3] for a specific matrix ``K``. If ``K`` (or scale parameter)
  changes, these polynomial coefficients have to be recalculated. The change
  can be detected by ``self.asymp_polynomial_hyperparam`` variable.

  Also, add the option of ``asymp_order`` to be 1 or 2. The polynomial of order
  1 is [a0, a1] and the polynomial of order 2 is [a0, a1, a2, a3].

  Add the option ``use_asymptotic``. If True, whenever eta is larger than the
  largest eigenvalue of K, computation of ell, ell_jacobian and ell_hessian
  should switch to their asymptotic relation. So create these functions:
  * ``asymp_likelihood``
  * ``asymp_likelihood_der1_eta``
  * ``asymp_likelihood_der2_eta``

  and in ``likelihood``, ``likelihood_jacobian``, and ``likelihood_hessian``,
  refer to these functions.

  Also, create a function ``asymp_der_1_zero`` to find zero of the polynomial.
  This is not used during optimization calls, rather, used for the initial
  hyperparam_guess.

------------------
measure imate time
------------------

* likelihood tic toc measure elapsed time and process time of computing imate.

-------------
imate configs
-------------

* pass imate config settings from cov object interface.

=====
Ideas
=====

* Does asymptotic relation exists for derivative w.r.t theta? If yes, its zero
  can be used to initialize theta_guess, similar to eta_guess.

======
Issues
======

* `_likelihood/_double_profile_likelihood.py`: In this method, using Jacobian
  seems to be strange. At small `theta`, the Jacobian matches the numerical
  evaluation of first derivative of likelihood. But for larger `theta`, the
  computed Jacobian seems to be *negative* of the first derivative. Thus,
  an optimization method that uses Jacobian cannot move an initial point, as
  in a direction that the function decreases, is Jacobian is wrongly positive!
  Hence the optimization iterations do not move the initial point to a lower
  function value and the final output will be the same as the initial point.
  To see this, use `plot=True`, and observe that the solid lines (Jacobian)
  is negative of dashed lines (numerical first derivative).
