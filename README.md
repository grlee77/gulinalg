gulinalg
========

Linear algebra functions as Generalized Ufuncs.

Notes about building
====================

This module is built using NumPy's configuration for LAPACK. This means that
you need a setup similar to the one used to build the NumPy you are using. If
you are building your own version of NumPy that should be the case.

OpenMP support
==============

A subset of functions currently have openMP support via a `workers` argument
that can be used to set the number of threads to use in the outer gufunc loop.

On windows MSVC-style flags will be set, otherwise GCC-style flags (-fopenmp)
are set. By default OpenMP is enabled, but if compilation of a simple test
function fails, OpenMP will be disabled,

The user can force OpenMP to always be disabled if desired by defining the
environment variable GULINALG_DISABLE_OPENMP.

On linux, linking against intel's OpenMP implementation instead of the GNU
implementation can be selected by defining GULINALG_INTEL_OPENMP. This will
cause libiomp5 and libpthread to be linked during compilation (instead of GCC's
libgomp). This should be done, for example, on MKL-based conda environments
where the intel-openmp package has been installed. For OpenBLAS-based conda
environments, the GULINALG_INTEL_OPENMP variable should not be defined.

If Intel's icc compiler is being used instead of gcc, the user should define
the GULINALG_USING_ICC environment variable. Use of icc on windows systems is
not currently supported.

Build Status
============

Travis CI: [![Build Status](https://travis-ci.org/Quansight/gulinalg.svg?branch=master)](https://travis-ci.org/Quansight/gulinalg)
