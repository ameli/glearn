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

from ._full_likelihood import FullLikelihood
from ._profile_likelihood import ProfileLikelihood
from ._double_profile_likelihood import DoubleProfileLikelihood


# ==========
# likelihood
# ==========

def likelihood(mean, cov, z, log_hyperparam=True, profile_hyperparam='var'):
    """
    Object factory method to create a new instance of the class.
    """

    # Set likelihood method depending on the type of profile.
    if profile_hyperparam == 'none':
        return FullLikelihood(mean, cov, z, log_hyperparam)
    elif profile_hyperparam == 'var':
        return ProfileLikelihood(mean, cov, z, log_hyperparam)
    elif profile_hyperparam == 'var_noise':
        return DoubleProfileLikelihood(mean, cov, z, log_hyperparam)
    else:
        raise ValueError('"profile_hyperparam" can be one of "none", ' +
                         '"var", or "var_noise".')
