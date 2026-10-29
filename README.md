# OpenMC Fundamental Testing

Fundamental and functional testing of the OpenMC for ITER non PIA applications.

This repository contains a collection of automated tests intended to provide
objective evidence for the verification and qualification of OpenMC for ITER
radiation transport applications.

> ITER_D_VP6DJL v2.1  
> *Instructions on qualification of radiation transport codes*

The tests are designed to be reproducible and suitable for automated execution
against current and future OpenMC releases.

## Scope

The current repository covers the fundamental tests required by the qualification
procedure, including:

| Requirement | Test                   | Status |
|-------------|------------------------|--------|
| 7.2.1.1   | Neutron and Photon Flux | TBD
| 7.2.1.2.1 | Nuclear Heat | TBD
| 7.2.1.2.2 | Tritium Production | Done
| 7.2.1.2.3 | Dose - SDDR | TBD
| 7.2.1.2.4 | Helium Production | TBD
| 7.2.1.2.5 | DPA | TBD
| 7.2.2.1   | Statistical error | Done
| 7.2.2.2   | Statistical error with weight windows | Done 
| 7.2.2.3   | Random number generator | Done
| 7.2.2.4   | Source sampling | Done
| 7.2.2.5   | Conservation of energy for nuclear reactions | Done

Additionally, a report folder is included.

The first routine to retrieve JADE results and post-process them has been developed - Tritium Production. Python script retrieves data from JADE-RAW and post processes it.


## Repository Structure

Each fundamental test is contained in its own directory.

A typical test has the structure:

```text
<test-name>/
├── run_test.py
└── ...

The repository also contains a report/ directory used to generate the qualification report:

report/
├── ...
├── detailed_results.tex
├── openmc_quickinstall.tex
├── publications.tex
└── ...

TO DO: add instructions on how to run the tests and compile the report.