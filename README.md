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
as a teaching device (and thus should be understandable by non-experts).
