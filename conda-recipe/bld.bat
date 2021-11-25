rem SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
rem SPDX-License-Identifier: BSD-3-Clause
rem SPDX-FileType: SOURCE
rem
rem This program is free software: you can redistribute it and/or modify it
rem under the terms of the license found in the LICENSE.txt file in the root
rem directory of this source tree.

@echo off
setlocal EnableDelayedExpansion

python -m pip install . -vv --no-binary :all:
