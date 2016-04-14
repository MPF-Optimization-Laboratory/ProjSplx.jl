# ProjSplx

[![Build Status](https://travis-ci.org/MPF-Optimization-Laboratory/ProjSplx.jl.svg?branch=master)](https://travis-ci.org/MPF-Optimization-Laboratory/ProjSplx.jl)
[![codecov.io](https://codecov.io/github/MPF-Optimization-Laboratory/ProjSplx.jl/coverage.svg?branch=master)](https://codecov.io/github/MPF-Optimization-Laboratory/ProjSplx.jl?branch=master)


Routines for projecting onto the simplex

    { x | sum(x) = t, x >= 0 }

or 1-norm ball

    { x |  ||x||_1 \le t }

Weighted versions also included.
