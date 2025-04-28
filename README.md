# 2D FFT-based Homogenization of Microstructures with Inclusions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

Numerical homogenization toolkit implementing FFT-based solvers for predicting effective elastic properties of composite materials with periodic microstructures. Implements both classical fixed-point and conjugate gradient-accelerated methods for solving periodic Lippmann-Schwinger equations in spectral domain.

## Key Features

- **Multi-solver Support**
  - Classical Fixed-Point FFT (Moulinec-Suquet,1998)
  - CG-accelerated FFT (Zeman et al. 2010)
  
- **Microstructure Generation**
  - Circular inclusions
  - Special-shaped inclusions (parametric)
  
- **Material Properties**
  - Isotropic linear elasticity
  - Arbitrary phase contrast ratios

 While developing this code, the following two central repositories have been referred to in addition to the original papers:
 
 - [FFT Microstructure.jl](https://github.com/Arvinth-shankar/FFT_Composite/blob/main/FFT_microstructure.jl)
 - [Standard FFT Linear Elastic](https://github.com/Firdes/FFT-based-Homogenization/blob/master/standardFFT-linear-elastic.py)
 
 **Original Papers:**  
- H. Moulinec, Pierre Suquet. A numerical method for computing the overall response of nonlinear composites with complex microstructure. *Computer Methods in Applied Mechanics and Engineering*, 1998, 157 (1-2), [10.1016/S0045-7825(97)00218-1](https://doi.org/10.1016/S0045-7825(97)00218-1)
- Zeman, J., Vondřejc, J., Novák, J., & Marek, I. (2010). Accelerating a FFT-based solver for numerical homogenization of periodic media by conjugate gradients. Journal of Computational Physics, 229(21), 8065-8071. https://doi.org/10.1016/j.jcp.2010.07.010
