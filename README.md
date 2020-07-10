# Case Study: Fairness and Generalizability of MIMIC-Trained Model 

This case study examines the fairness and generalizability of a MIMIC-trained model published in the peer-reviewed article on 
[Multitask learning and benchmarking with clinical time series data](https://www.nature.com/articles/s41597-019-0103-9) by Harutyunyan et al., 
with the code available [here](https://github.com/YerevaNN/mimic3-benchmarks).

The source code made available by Harutyunyan et al. has been updated and adapted to this analysis framework.
All code necessary to reconstruct the same analyses, including the Jupyter notebooks, can be
found in this repository.

This repository contains four main directories:

- `benchmarks`: Pipeline to construct in-hospital mortality (IHM) cohorts
    - *mimic*: Using MIMIC-III v1.4 relational database
    - *starr*: Using STARR_DE relational database
    - shared scripts
- `models`: Training, testing and evaluation protocols
- `notebooks`: Relevant Jupyter notebooks 
- `slurm_jobs`: SLURM scripts for training the models





