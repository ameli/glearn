*****
Notes
*****

* `correlation` > `_generate_sparse_correlation.pyx` > add geenration of
  the derivatives of correlaton (jacobian and hessian). See
  `_generate_dense_correlation.pyx`.

====
ToDo
====

* Add prior distributions for scale parameters :math:`\theta`.
* in ``_profile_likelihood.pyx``, in the function ``find_optimal_sigma``,
  sime variables are computed before this function is called, like ``Y``,
  ``w``, etc. Passing these can reduce the runtime.
* in profile likelihood, the last term of lp is a constant. zMz deviced by
  the optimal value of sigma**2 is constant. This is only for profile
  likelihood where optimal value of sigma is used.
