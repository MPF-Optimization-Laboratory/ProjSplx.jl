# ProjSplx

[![Build Status](https://travis-ci.org/MPF-Optimization-Laboratory/ProjSplx.jl.svg?branch=master)](https://travis-ci.org/MPF-Optimization-Laboratory/ProjSplx.jl)
[![codecov.io](https://codecov.io/github/MPF-Optimization-Laboratory/ProjSplx.jl/coverage.svg?branch=master)](https://codecov.io/github/MPF-Optimization-Laboratory/ProjSplx.jl?branch=master)


This package provides routines for projecting onto the simplex

    { x | sum(x) = t, x≥0 }

and the 1-norm ball with radius t:

    { x |  ||x||_1 ≤ t }

Weighted versions also included.

The provided algorithms are based on the projection algorithm devised by

    [van den Berg and Friedlander (2008), "Probing the Pareto frontier for basis pursuit solution", SIAM J Optimization 2(31), 2008](https://friedlander.io/files/pdf/2008BergFriedlander.pdf)
