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

| Requirement | Test |
|-------------|------|
| 7.2.2.1 | Statistical error |
| 7.2.2.2 | Statistical error with weight windows |
| 7.2.2.3 | Random number generator |
| 7.2.2.4 | Source sampling |
| 7.2.2.5 | Conservation of energy for nuclear reactions |

Additionally, a report folder is included.

In the future routines to automatically retrieve JADE results and post-process them in to the format needed by the report will be added.

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