---
title: 'Pippin'
tags:
  - Python
  - pipeline
  - supernova
  - cosmology
authors:
  - name: Samuel Hinton
    orcid: 0000-0003-2071-9349
    affiliation: 1
affiliations:
  - name: University of Queensland
    index: 1
date: 12 Feb 2020
bibliography: paper.bib
---

# Summary

Pippin is a python pipeline for supernova cosmology analysis. It is designed to allow
a user to specify a cosmological analysis via a configuration file, and then
run that analysis, start to finish, in a single execution.

To this end, Pippin interfaces 
with multiple external programs in the execution of its derived tasks.
These tasks include data preparation, supernova simulation using SNANA [@snana], 
light curve fitting tasks, machine learning classification of transients, 
computing the required bias corrections and feeding the results
into CosmoMC [@cosmomc]. The MCMC outputs are then analysed in ChainConsumer [@ChainConsumer].

Additonal plots are performed via the matplotlib library [@matplotlib], and 
makes use of various numpy [@numpy], scipy [@scipy] and pandas [@pandas] functions. 

Code archives can be found on Zenodo at [@zenodo] and any
bugs or feature requests can be opened as issues on the Github
development page [@github].

# References
