# Exploring the Non-zero Eccentricity of WASP-107~b

[![DOI](https://zenodo.org/badge/DOI/10.XXXX/zenodo.XXXXXX.svg)](https://doi.org/XXXX) <!-- ADD ACTUAL DOI BADGE -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the data and analysis code for the paper:

**Confirmation of Non-zero Eccentricity for the Super-Puff Exoplanet WASP-107~b using JWST Occultation Observation**

## 📁 Repository Contents
- `0-data/`: Raw data and files for data processing
  - `0-data/jwst/`: Configuration files for JWST data reduction pipelines (`Eureka!`, `exoTEDRF`)
  - `0-data/tess/`: TESS light curves processed with the `SPOC` pipeline

- `1-transit analysis/`: Transit light curve modeling and timing analysis
  - `1-transit analysis/LC/`: Light curves for individual transits
  - `1-transit analysis/TTV/`: Transit timing variations analyzed with linear and orbital decay models

- `2-joint fitting/`: Combined global fit using eclipse, radial velocity, and transit data

- `parameters.json`: Planetary system parameters used in analysis