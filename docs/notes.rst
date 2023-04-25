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

=============
Documentation
=============

Things yet remained in the documentation to be completed:

* covariance class
* Gaussian process class
* A few more tutorials in jupyter notebook
* performance/.. is empty
* tutorials/gpu.rst: the examples are for imate and have not been updated to be
  the examples for glearn.. Once GaussianProcess class is finished, update the
  usage examples in gpu.rst.
* In index.rst, I changed the card for "Performance" to "Publications", since
  the content for performance are not ready.
* overview.rst has to be completely rewritten and to be referenced in index.rst.
* ISSUE: in /docs/source/tutorials/gu.rst, in section "3.5. Run GLearn
  Functions on GPU", all context there is for imate instead of glearn. Provide
  example of running glearn on GPU, and create a sample output txt file in
  /docs/source/_statis/data/
