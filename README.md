# 2D FFT-based Homogenization of Microstructures with Inclusions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

Numerical homogenization toolkit implementing both Fixed-Point and Conjugate Gradient (CG) accelerated FFT-based solvers for linear elastic composite materials with periodic microstructures.

<img src="images/stress_field.png" width="45%"> <img src="images/microstructure.png" width="45%">

## Key Features

- **Multi-solver Support**
  - Classical Fixed-Point FFT (Moulinec-Suquet)
  - CG-accelerated FFT (Zeman et al. 2010)
  
- **Microstructure Generation**
  - Circular inclusions
  - Special-shaped inclusions (parametric)
  - Custom phase distributions
  
- **Material Properties**
  - Isotropic linear elasticity
  - Arbitrary phase contrast ratios
  - Automatic reference medium computation

- **Advanced Features**
  - Green's operator in Fourier space
  - Spectral convergence monitoring
  - Stress/strain visualization
  - Effective property computation

