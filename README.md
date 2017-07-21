# MC analysis of BA algorithms


Building on the per-param and data stuff is a full MC framework for estimating the posterior. 

The focus here is on running tons of simulations varying parameters and observations in attempt to characterise algorithm consistency (Pb) and performance (Ps) given parameters and data.

There are many information metrics we can determine from Pb and Ps related to properties of the input data. 


Given the high dimensionality of the systems (200x200x366) a simple MC approach is taken. Further for the data (for now) we assume correlated data gaps and gaussian noise... To analyse the posterior conditional on the data we need to derive some useful properties of the data. Some initial suggestions seem to be the number of observations, data noise (assumed iid gaussian) and days before and after fire to an observation. These are essentially summary statistics of the properties of the data distribution P(D) in terms of sparsity and error. 

For individual pixels it is possible to simulate a full uncertainty tracking but this may require too many runs given the dimensionality of the space-time problem.

