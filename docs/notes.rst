*****
Notes
*****

* In profile likelihood, the last term of `lp` is a constant. `zMz`, at the
  optimal value of sigma**2, is constant. This is only for profile likelihood
  where optimal value of sigma is used.
* For initial guesses of sigma and sigma0 hypepraram, assume sigma is zero, and
  use eta=inf to find optimal sigma0 as a hyperparam.

====
ToDo
====

--------------------
Asymptotic Relations
--------------------

* Add the option ``use_asymptotic``. If True, whenever eta is larger than the
  largest eigenvalue of K, computation of ell, ell_jacobian and ell_hessian
  should switch to their asymptotic relation. So create these functions:
      + ``asymp_likelihood``
      + ``asymp_likelihood_der1_eta``
      + ``asymp_likelihood_der2_eta``
  and in ``likelihood``, ``likelihood_jacobian``, and ``likelihood_hessian``,
  refer to these functions.

-------------
imate configs
-------------

* pass imate config settings from cov object interface.

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
