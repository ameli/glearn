*****
Notes
*****

* In profile likelihood, the last term of `lp` is a constant. `zMz`, at the
  optimal value of sigma**2, is constant. This is only for profile likelihood
  where optimal value of sigma is used.

====
ToDo
====

---------------
Store Variables
---------------

* In full likelihood method, separate Jacobian and Hessian to smaller functions
  for the derivative of each component.
* In profile and full method, avoid recomputing ``Y``, ``Binv``, and ``Mz`` by
  storing these variables as attribute.
* In ``_profile_likelihood.pyx``, in the function ``find_optimal_sigma``,
  sigma variables are computed before this function is called, like ``Y``,
  ``w``, etc. Passing these can reduce the runtime.

-----
Prior
-----

* Add prior distributions for scale parameters :math:`\theta`. This requires
  storing function and its derivatives in an instance of likelihood class.
* A class like ``posterior`` might be needed, with a method ``maximize``, in
  which we modify a derived class from the ``scipy.optimize.minimize``.

-------
Predict
-------

* Add ``predict`` to ``gaussian_process`` class.

-----------------
Hutchinson Method
-----------------

* In imate package, complete Hutchinson's method to compute trace of
  Ainv * B1 * Ainv * B2.
* In profile and full likelihood methods, implement Hutchinson's method to
  compute the trace of some matrices related to the derivative of scale.

--------------------
Asymptotic Relations
--------------------

* Use asymptotic derivatives w.r.t eta to find ``eta_guess`` (used in
  ``hyperparam_guess``) when user does not provide a guess for eta.

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
