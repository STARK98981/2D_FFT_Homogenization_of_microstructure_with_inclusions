import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

# ---------------------
# Geometry and Discretization
# ---------------------
ndim = 2
Nx = Ny = N = 129  # grid dimensions
Ngrids = Nx * Ny
phase_contrast = 0.5
fibre_volume = 0.2

# ---------------------
# Field Initialization: Prescribed Strain
# ---------------------
Epsilon_prescribed = np.zeros((ndim, ndim))
Epsilon_prescribed[0, 0] = 0.1
Epsilon_prescribed[1, 1] = 0.1

# ---------------------
# Tensor Helpers
# ---------------------
delta = lambda i, j: 1.0 if i == j else 0.0
ddot_42 = lambda A4, B2: np.einsum('ijkl,lk->ij', A4, B2)
ddot_21 = lambda A2, B1: np.einsum('ij,j->i', A2, B1)
ddot_12_transposed = lambda A1, B2: np.einsum('j,ji->i', A1, B2)
# ---------------------
# Material Properties and Stiffness Tensors
# ---------------------
# Phase 2 (fiber) parameters
Lambda_f = 1.0
mu_f = 1.0

# Phase 1 (matrix) parameters
Lambda_m = phase_contrast * Lambda_f
mu_m = phase_contrast * mu_f

# Prebuild 4th-order stiffness tensors for fiber and matrix
C_f_tensor = np.zeros((ndim, ndim, ndim, ndim))
C_m_tensor = np.zeros((ndim, ndim, ndim, ndim))
for i in range(ndim):
    for j in range(ndim):
        for k in range(ndim):
            for l in range(ndim):
                C_f_tensor[i, j, k, l] = Lambda_f * delta(i, j) * delta(k, l) + \
                                         mu_f * (delta(i, k) * delta(j, l) + delta(i, l) * delta(j, k))
                C_m_tensor[i, j, k, l] = Lambda_m * delta(i, j) * delta(k, l) + \
                                         mu_m * (delta(i, k) * delta(j, l) + delta(i, l) * delta(j, k))

# ---------------------------------------------------------------
# Build Phase Distribution for a circular inclusion at the centre
# ---------------------------------------------------------------
phase = np.zeros((N, N))  

# Functions for defining different inclusions
def Circular_inclusion():
    R2 = fibre_volume * N**2 / np.pi
    for x in range(N):
        for y in range(N):
            if ((x - N//2)**2 + (y - N//2)**2) <= R2:
                phase[x, y] = 1  

def Special_inclusion():  
    R2 = N * np.sqrt(2 * fibre_volume / (np.pi))
    for x in range(N):
        for y in range(N):
            theta = np.arctan2(y -  N//2, x -  N//2)
            if np.sqrt(((x - N//2)**2 + (y - N//2)**2)) <= R2*np.abs(np.cos(2 * theta)):
                phase[x, y] = 1

# Choose function depending on the inclusion desired
Circular_inclusion()
#Special_inclusion()

# Construct C_mat: assign stiffness tensor at each grid point
C_mat = np.empty((Nx, Ny, ndim, ndim, ndim, ndim))
for x in range(Nx):
    for y in range(Ny):
        if phase[x, y] == 1:
            C_mat[x, y] = C_f_tensor
        else:
            C_mat[x, y] = C_m_tensor

"""
# Plot the microstructure using a heatmap.
plt.figure(figsize=(6, 6))
plt.imshow(phase, cmap=cmr.redshift, origin='lower')
plt.colorbar(label='Phase (1: fiber, 2: matrix)')
plt.title('Phase Distribution: Fiber in Matrix')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""

# ---------------------
# Green's Operator in Fourier Space
# ---------------------
dx = 1.0
Lambda_0 = (Lambda_m + Lambda_f) / 2
mu_0 = (mu_m + mu_f) / 2
const_Green = (Lambda_0 + mu_0) / (mu_0 * (Lambda_0 + 2 * mu_0))

# Create 2D frequency grid using unitless frequencies.
fx = (np.fft.fftfreq(Nx, d=dx))
fy = (np.fft.fftfreq(Ny, d=dx))
FX, FY = np.meshgrid(fx, fy, indexing='ij')
freq = np.stack((FX, FY), axis=-1)  # shape (Nx, Ny, 2)

# Build the Green's operator Gamma (shape: (Nx, Ny, 2, 2, 2, 2))
Gamma = np.zeros((Nx, Ny, ndim, ndim, ndim, ndim))
for x in range(Nx):
    for y in range(Ny):
        q = freq[x, y]
        qnorm2 = np.dot(q, q)
        if qnorm2 == 0:
            continue
        for i in range(ndim):
            for j in range(ndim):
                for k in range(ndim):
                    for h in range(ndim):
                        term1 = ( delta(k, i)*q[h]*q[j] + 
                                  delta(h, i)*q[k]*q[j] + 
                                  delta(k, j)*q[h]*q[i] + 
                                  delta(h, j)*q[k]*q[i] ) / (4 * mu_0 * qnorm2)
                        
                        term2 = const_Green * (q[i]*q[j]*q[k]*q[h]) / qnorm2**2
                        Gamma[x, y, i, j, k, h] = term1 - term2

# ---------------------
# Initialize Local Fields: Strain and Stress
# ---------------------
Epsilon = np.zeros((Nx, Ny, ndim, ndim))
Sigma = np.zeros((Nx, Ny, ndim, ndim))
for x in range(Nx):
    for y in range(Ny):
        Epsilon[x, y] = Epsilon_prescribed
        Sigma[x, y] = ddot_42(C_mat[x, y], Epsilon_prescribed)

def average_stress(sigma):
    # Average the stress over the spatial grid (axes 0 and 1)
    return np.mean(sigma, axis=(0,1))


initial_sigma_avg = average_stress(Sigma)
print("Initial average stress (before iteration):")
print(initial_sigma_avg)

# ---------------------
# Iterative FFT-based Solver
# ---------------------
Sigma_hat = np.zeros((Nx, Ny, ndim, ndim), dtype=np.complex64)
Epsilon_hat = np.zeros((Nx, Ny, ndim, ndim), dtype=np.complex64)

# Define the index corresponding to zero frequency
c = 0

def calculate_denominator(sigma_fourier, c):
    # Use the zero-frequency value
    return np.sqrt(np.sum(np.abs(sigma_fourier[c, c])**2))

def calculate_numerator(sigma_fourier, freq):
    result = 0.0
    for x in range(Nx):
        for y in range(Ny):
            temp = ddot_12_transposed(freq[x, y],sigma_fourier[x, y])
            result += np.sum(np.abs(temp)**2)
    return np.sqrt(result / Ngrids)

iter = 1
max_iter = 1000
tolerance = 1e-6

while True:
    # Forward FFT: compute Fourier transform of stress field for each component
    for i in range(ndim):
        for j in range(ndim):
            Sigma_hat[:,:,i,j] = np.fft.fft2(Sigma[:,:,i,j])
    
    numerator = calculate_numerator(Sigma_hat, freq)
    denominator = calculate_denominator(Sigma_hat, c)
    error = numerator / denominator
    print(f"Iteration {iter}, error = {error:.3e}")
    if error < tolerance:
        break

    
    # Strain update: update the Fourier-transformed strain field using the Green operator
    Epsilon_hat = np.zeros((Nx, Ny, ndim, ndim), dtype=np.complex64)
    for i in range(ndim):
        for j in range(ndim):
            Green_convoluted_Sigma = np.zeros((Nx, Ny), dtype=np.complex64)
            for k in range(ndim):
                for l in range(ndim):
                    Green_convoluted_Sigma += Gamma[:,:,i,j,k,l] * Sigma_hat[:,:,k,l]
            # Update rule: subtract convolution from current Fourier strain.
            Epsilon_hat[:,:,i,j] = np.fft.fft2(Epsilon[:,:,i,j]) - Green_convoluted_Sigma
    


    # Enforce the prescribed mean strain E for @ Zero Frequency
    for i in range(ndim):
        for j in range(ndim):
            # Forward FFT is rather direct (unscaled) amd Inverse FFT is scaled by Nx*Ny
            Epsilon_hat[c, c, i, j] = Epsilon_prescribed[i, j] * Ngrids
            # Inverse FFT to update strain in real space
            Epsilon[:,:,i,j] = np.real(np.fft.ifft2(Epsilon_hat[:,:,i,j]))
    
    # Update stress field via constitutive relation: Sigma = C_mat : Epsilon
    for x in range(Nx):
        for y in range(Ny):
            Sigma[x,y] = ddot_42(C_mat[x,y], Epsilon[x,y])
    
    if iter >= max_iter:
        print("\nMaximum iterations reached!")
        break
    iter += 1

if iter < max_iter:
    print("\nConverged in", iter, "iterations")
else:
    print("\nDid not converge within the maximum number of iterations.")

# ---------------------
# Postprocessing: Compute average stress and plot stress components
# ---------------------


sigma_avg = average_stress(Sigma)
print("Average stress:")
print(sigma_avg)

# Define spatial grid for plotting (assuming domain [0,1] in each direction)
x_grid = np.linspace(0, 1, Nx)
y_grid = np.linspace(0, 1, Ny)

def contour_plot_sigma(ax, sigma, component, title):
    i, j = component
    cf = ax.contourf(x_grid, y_grid, sigma[:,:,i,j], levels=100, cmap=cmr.redshift)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(cf, ax=ax)

fig, axs = plt.subplots(2, 2, figsize=(10,10))
components = [(0,0), (0,1), (1,0), (1,1)]
titles = ["σ₁₁", "σ₁₂", "σ₂₁", "σ₂₂"]

for ax, comp, tit in zip(axs.flat, components, titles):
    contour_plot_sigma(ax, Sigma, comp, tit)

plt.tight_layout()
plt.show()
