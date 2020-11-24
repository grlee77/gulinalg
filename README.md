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

If Intel's OpenMP library should be linked to, the user should specify the
environment variable GULINALG_INTEL_OPENMP. This will cause libiomp5 to be
linked during compilation (instead of GCC's libgomp).

Build Status
============

Travis CI: [![Build Status](https://travis-ci.org/Quansight/gulinalg.svg?branch=master)](https://travis-ci.org/Quansight/gulinalg)
