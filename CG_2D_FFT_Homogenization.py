import numpy as np
import matplotlib.pyplot as plt
#import cmasher as cmr

def solve_cg_fft(E_presc):

    # ---------------------
    # Geometry and Discretization
    # ---------------------
    ndim = 2
    Nx = Ny = N = 129  # grid dimensions
    Ngrids = Nx * Ny
    phase_contrast = 0.6
    fibre_volume = 0.2

    # ---------------------
    # Field Initialization: Prescribed Strain
    # ---------------------
    #Epsilon_prescribed = np.zeros((ndim, ndim))
    #Epsilon_prescribed[0, 0] = 0.1
    #Epsilon_prescribed[1, 1] = 0.1

    Epsilon_prescribed = E_presc

    # ---------------------
    # Tensor Helpers
    # ---------------------
    delta = lambda i, j: 1.0 if i == j else 0.0
    ddot_42 = lambda A4, B2: np.einsum('ijkl,lk->ij', A4, B2)

    # ---------------------
    # Material Properties and Stiffness Tensors
    # ---------------------
    Lambda_f = 1.0
    mu_f     = 1.0
    Lambda_m = phase_contrast * Lambda_f
    mu_m     = phase_contrast * mu_f

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

    phase = np.zeros((Nx, Ny))  

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
    plt.colorbar(label='Phase (1: fiber, 0: matrix)')
    plt.title('Phase Distribution: Fiber in Matrix')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    """


    # ---------------------
    # Reference Medium and δC
    # ---------------------
    Lambda_0 = 0.5*(Lambda_f + Lambda_m)
    mu_0     = 0.5*(mu_f     + mu_m)

    C0 = np.zeros((ndim,ndim,ndim,ndim))
    for i in range(ndim):
        for j in range(ndim):
            for k in range(ndim):
                for l in range(ndim):
                    C0[i, j, k, l] = Lambda_0 * delta(i, j) * delta(k, l) + \
                                            mu_0 * (delta(i, k) * delta(j, l) + delta(i, l) * delta(j, k))

    deltaC = C_mat - C0   # contrast field

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
    # FFT-based Operators
    # ---------------------
    def apply_Astar(eps_star):
        # (I - Γ₀ δC) · eps_star
        tau    = np.einsum('xyijkl,xykl->xyij', deltaC, eps_star)
        tau_hat= np.fft.fft2(tau, axes=(0,1))
        conv_hat = np.zeros_like(tau_hat, dtype=complex)
        for i in range(ndim):
            for j in range(ndim):
                for k in range(ndim):
                    for l in range(ndim):
                        conv_hat[:,:,i,j] += Gamma[:,:,i,j,k,l]*tau_hat[:,:,k,l]
        conv = np.real(np.fft.ifft2(conv_hat, axes=(0,1)))
        return eps_star - conv

    def apply_M_inv(r):
        # M^{-1} r = - Γ₀ * r
        r_hat = np.fft.fft2(r, axes=(0,1))
        s_hat = np.zeros_like(r_hat, dtype=complex)
        for i in range(ndim):
            for j in range(ndim):
                for k in range(ndim):
                    for l in range(ndim):
                        s_hat[:,:,i,j] -= Gamma[:,:,i,j,k,l]*r_hat[:,:,k,l]
        return np.real(np.fft.ifft2(s_hat, axes=(0,1)))

    # ---------------------
    # Build RHS: f = -Γ₀[δC : E_presc]
    # ---------------------
    E_grid = np.zeros((Nx,Ny,ndim,ndim))
    for x in range(Nx):
        for y in range(Ny):
            E_grid[x,y] = Epsilon_prescribed

    f_rhs = -(apply_Astar(E_grid) - E_grid)

    # ---------------------
    # Iterative CG-FFT Solver
    # ---------------------
    max_iter = 500
    tol      = 1e-7

    eps_star  = np.zeros_like(f_rhs)
    r         = f_rhs.copy()
    s         = apply_M_inv(r)
    d         = s.copy()
    delta_old = np.sum(r * s)

    for iter in range(1, max_iter+1):
        Ad    = apply_Astar(d)
        denom = np.sum(d * Ad)
        if abs(denom) < 1e-14:
            raise RuntimeError("Breakdown in CG: denom≈0")
        alpha = delta_old / denom

        eps_star += alpha * d
        r        -= alpha * Ad

        res_norm = np.linalg.norm(r)/np.linalg.norm(f_rhs)
        #print(f"Iteration {iter}, residual = {res_norm:.2e}")
        if res_norm < tol:
            #print(f"\nConverged in {iter} iterations.")
            break

        s_new     = apply_M_inv(r)
        delta_new = np.sum(r * s_new)
        beta      = delta_new / delta_old

        d         = s_new + beta * d
        s         = s_new
        delta_old = delta_new
    else:
        print("\nDid not converge within the maximum iterations.")

    # ---------------------
    # Postprocessing
    # ---------------------
    Epsilon = np.zeros_like(E_grid)
    Sigma   = np.zeros_like(E_grid)

    for x in range(Nx):
        for y in range(Ny):
            Epsilon[x,y] = Epsilon_prescribed + eps_star[x,y]
            Sigma[x,y]   = ddot_42(C_mat[x,y], Epsilon[x,y])

    # Compute average stress
    Sigma_avg = np.mean(Sigma, axis=(0,1))
    #print("\nAverage stress (σ):", Sigma_avg)
    return Sigma_avg

    """
    # ---------------------
    # Plotting
    # ---------------------
    x_grid = np.linspace(0,1,Nx)
    y_grid = np.linspace(0,1,Ny)

    def contour_plot_sigma(ax, component, title):
        i,j = component
        cf = ax.contourf(x_grid, y_grid, Sigma[:,:,i,j], levels=50, cmap=cmr.redshift)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        plt.colorbar(cf, ax=ax)

    fig, axs = plt.subplots(2,2, figsize=(10,8))
    components = [(0,0),(0,1),(1,0),(1,1)]
    titles     = ["σ₁₁","σ₁₂","σ₂₁","σ₂₂"]

    for ax, comp, tit in zip(axs.flat, components, titles):
        contour_plot_sigma(ax, comp, tit)

    plt.tight_layout()
    plt.show()
    """
# ----------------------------------------
#  Compute homogenized material stiffness matrix by 3 independent loads
# ----------------------------------------
# Voigt ordering [11,22,12]:
macros = [
    np.array([[1,0],[0,0]]),   # E11=1
    np.array([[0,1],[1,0]]),   # E12=1 
    np.array([[0,0],[0,1]]),   # E22=1
]
Sig_avgs = [solve_cg_fft(Eb) for Eb in macros]

# Assemble 3×3 Voigt matrix C_homog
C_hom = np.zeros((3,3))

# E11 response (column 0)
C_hom[:, 0] = [Sig_avgs[0][0,0], Sig_avgs[0][1,1], Sig_avgs[0][0,1]]

# E22 response (column 1)
C_hom[:, 1] = [Sig_avgs[2][0,0], Sig_avgs[2][1,1], Sig_avgs[2][0,1]]

# E12 response (column 2, scaled by 0.5)
shear_avg = Sig_avgs[1]
C_hom[:, 2] = [shear_avg[0,0]/2, shear_avg[1,1]/2, shear_avg[0,1]/2]

print("\nHomogenized stiffness C_hom (Voigt 11,22,12):")
print(C_hom)