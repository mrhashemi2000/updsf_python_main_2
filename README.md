# updsf_python_main_2

Multiphysics Simulations of Chemical Selection for DNA in Hydrothermal Environments: Integrating Uracil Purge, Ribose-to-Deoxyribose Conversion, and Polymer Selection in Pore Networks


https://doi.org/10.5281/zenodo.20733760
https://doi.org/10.5281/zenodo.20759622
https://doi.org/10.5281/zenodo.20771213
https://doi.org/10.5281/zenodo.18594133

Author: Seyed Mohammad Reza Hashemi (Reza Hashemi) 

Overview
UPDSF is a comprehensive computational framework designed to simulate the RNA-to-DNA transition in prebiotic hydrothermal environments. It integrates three core stochastic engines:

1. VSSUF (Vent Stochasticity Seals Uracil's Fate): Simulates selective monomer hydrolysis.
2. RD Converter: Models the clay-protected conversion of ribose to deoxyribose.
3. HYDRA (Hydrothermal Dynamics of Replicative Amplification): Simulates polymer-level selection within pore networks using thermophoresis and diffusion.

Key Features in v1.2
- Scientifically Validated Rates: Kinetic parameters derived from literature (Kawamura 2003, Marrone 2010).
- High Performance: Optimized using Numba (JIT compilation) for Monte Carlo simulations.
- Robustness:* Added division-by-zero protections and corrected distance transform algorithms.

Installation
git clone https://github.com/mrhashemi2000/UPDSF-Framework.git
pip install -r requirements.txt

Quick Start
Run the full 72-hour simulation demo:
python updsf_main.py

Citation
If you use this framework in your research, please cite it as:
Hashemi, S. R. (2026). Unified Prebiotic DNA Selection Framework (UPDSF) v1.2. https://doi.org/10.5281/zenodo.20825578
