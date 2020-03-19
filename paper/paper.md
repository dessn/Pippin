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
  - name: Dillon Brout
    orcid: 0000-0001-5201-8374
    affiliation: "2, 3"
affiliations:
  - name: University of Queensland
    index: 1
  - name: University of Pennsylvania
    index: 2
  - name: NASA Einstein Fellow
    index: 3
  
date: 12 Feb 2020
bibliography: paper.bib
---

# Summary

Pippin is a python pipeline for supernova cosmology analysis. It is designed to allow
a user to specify a cosmological analysis via a configuration file, and then
run that analysis, start to finish, in a single execution. Whilst other tools exist to perform
various aspects of supernova cosmology analyses independently, connecting the required tasks has so far been 
a manual process. For the DES-3YR analysis [@des3yr], we implemented a basic bash pipeline to reduce 
manual labour. For the DES-5YR analysis, which greatly increases complexity, we implement this pipeline to increase
researcher productivity.

To this end, Pippin interfaces 
with multiple external programs in the execution of its derived tasks.
These tasks include data preparation, supernova simulation using SNANA [@snana], 
light curve fitting tasks, machine learning classification of transients, 
computing the required bias corrections and feeding the results
into CosmoMC [@cosmomc]. The MCMC outputs are then analysed in ChainConsumer [@ChainConsumer].

This functionality allows Pippin to run end-to-end, taking telescope data as input and producing cosmological
constraints in under a day, with no input required from the user past the initial configuration file. It is 
currently in use in multiple projects, including investigating the impact of classifiers on cosmology, determining
template tuning of the supernova core-collapse rate, assessing the impact of spectral features on classification, 
and determining the reduction in cosmological bias from improved calibration of the SALT2 supernova model.


Additonal plots are performed via the matplotlib library [@matplotlib], and 
makes use of various numpy [@numpy], scipy [@scipy] and pandas [@pandas] functions. Several 
machine learning functions make use of scikit-learn [@sklearn].

Any bugs or feature requests can be opened as issues on the GitHub
[development page](https://github.com/Samreay/Pippin/issues) [@github].

# References
