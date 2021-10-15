# Fish Fry

Fish Fry is a benchmark that solves a 3D Poisson problem using MPI communication and GPU computation.
In particular, it solves the equation

&Del;<sup>2</sup>*&phi;* = *f*,

where

*&phi;* = *e*<sup>sin *x*</sup> cos *x* *e*<sup>cos *y*</sup> sin *y* *e*<sup>cos *z*</sup> cos *z*

and

*f* = –*e*<sup>sin *x*</sup> sin *x* cos *x* (sin *x* + 3) *e*<sup>cos *y*</sup> sin *y* (sin<sup>2</sup> *y* – 3 cos *y* – 1) *e*<sup>cos *z*</sup> (3 sin<sup>2</sup> *z* – cos<sup>3</sup> *z* – 1).

The benchmark sets *f*, solves for *&phi;*, and compares *&phi;* against the analytic solution.
For each run, the benchmark prints the runtime, the rate of points per second, and the
*L*<sub>1</sub>, *L*<sub>2</sub>, and *L*<sub>&infin;</sub> error norms for the computed *&phi;*.
It then prints the average time and rate across the runs.

The benchmark takes six or seven arguments as input:
* the number of MPI tasks in each dimension,
* the number of points in each dimension,
* and, optionally, the number of runs of the benchmark, where the default is 1.
