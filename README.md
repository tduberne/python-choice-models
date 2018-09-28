Choice Model Estimation in Python
=================================

This repository contains experiments around the estimation of choice models in Python.

Why?
----

There exist a variety of packages and scripts to estimate choice models,
but none that fitted perfectly my needs. I wanted something that is:

- native python (and can thus be used in notebooks)
- expressive
- flexible

Idea
----

The idea is mostly to provide a layer of abstraction over Scipy's optimization routines,
that allows to specify the log-likelihood of statistical models in an expressive way,
focusing on the usecase of discrete choice models (but in theory applicable to any maximum-likelihood
estimation).

I also try to keep the code as simple as possible, as the idea is that the code should also be usable
as a teaching device (and thus should be understandable without too much Python or software engineering
background).

The main features it brings are:

- possibility to access parameters by name from the optimized function (likelihood)
- methods to compute estimated standard errors of estimators, covariance of estimators, etc.
- possibility to define names, starting values and contraints on parameters in one statement

What is not planned is:

- to implement automatic derivation of analytical gradient
- to be complete in terms of the kind of models that are provided out-of-the-box
  (additional models are however easy to implement by specifying the likelihood function)

How to Use
----------

For the moment, the easiest way to use is to copy the `*.py` files from this repository in
the same directory as your notebooks. This also fits well in teaching.
