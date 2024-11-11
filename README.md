# Vulnerability-Toolkit

![logo](https://github.com/mouayed-nafeh/random/assets/149155077/024e652c-8bc3-41c1-b2fc-0ae1e9f5816a)

## 🔎 Overview 

The **Vulnerability-Toolkit** is an open source library that provides modelling of multi-degree-of-freedom systems and assessment via nonlinear time-history analyses for vulnerability and risk calculations. The **Vulnerability-Toolkit** is developed by the **[GEM](http://www.globalquakemodel.org)** (Global Earthquake Model) Foundation and its collaborators.

DOI: TBD

## 🛠️ Current Features

* MDOF Modelling: Model single- and multi-degree-of-freedom system using low-level information (e.g., number of storeys, first-mode transformation factor, SDoF- or storey-based force-deformation relationships);
* Modal analysis: Estimate periods of vibration and modal shapes;
* Static analysis: Perform gravity, static and cyclic pushover analyses;
* Dynamic analysis: Perform cloud analysis;
* Regression analysis: Perform regression analysis on cloud analysis data to characterise EDP|IM relationship;
* Fragility analysis: Calculate median seismic intensities, associated dispersion (i.e., record-to-record and modelling uncertainties) and the corresponding probabilities of damage based on demand-based thresholds;
* Vulnerability analysis: Perform fragility function convolution with with consequence models  for structural, and building contents and region-specific non-structural storey-loss functions;
* Plotting: Plot analysis outputs such as model overview, cloud analysis results, demand profiles (i.e., peak storey drifts and peak floor acceleration along the height of the model), fragility functions. Additionally, it is possible to animate the MDoF considering a single run;

## 📚 Documentation

TBD

# 🌟 Contributors

Contributors are gratefully acknowledged and listed in CONTRIBUTORS.txt. 

# © License
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# 🤔 Frequently asked questions

### How to contribute?

You can follow the instructions indicated in the [contributing guidelines](./contribute_guidelines.md). (Work-In-Progress)

### Which version am I seeing? How to change the version?

By default, you will see the files in the repository in the `main` branch. Each version of the model that is released can be accessed is marked with a `tag`. By changing the tag version at the top of the repository, you can change see the files for a given version.

Note that the `main` branch could contain the work-in-progress of the next version of the model.

# References
[^1]: Nafeh, A.M.B., Al-Jawhari, K., Silva, V, (202X). Title Pending. Journal
[^2]: Nafeh, A.M.B., Al-Jawhari, K., Silva, V, (202X). Title Pending. COMPDYN
