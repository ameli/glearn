*****
Notes
*****

* `correlation` > `_generate_sparse_correlation.pyx` > add generation of
  the derivatives of correlation (Jacobian and Hessian). See
  `_generate_dense_correlation.pyx`.
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
* In profile likelihood, the last term of `lp` is a constant. `zMz`, at the
  optimal value of sigma**2, is constant. This is only for profile likelihood
  where optimal value of sigma is used.

====
ToDo
====

* Fix derivatives w.r.t `theta` in the full likelihood class.
* Add prior distributions for scale parameters :math:`\theta`. This requires
  storing function and its derivatives in an instance of likelihood class.
* Make logarithmic optimization of `eta` and `theta`. This requires storing
  function and its derivatives as member data of likelihood class.
* Add `predict` to `gaussian_process` class.
* In ``_profile_likelihood.pyx``, in the function ``find_optimal_sigma``,
  sigma variables are computed before this function is called, like ``Y``,
  ``w``, etc. Passing these can reduce the runtime.
