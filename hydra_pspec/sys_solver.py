# Function definitions for GCR solver

import numpy as np
from math import pi as pi
import matplotlib.pylab as plt
from scipy.linalg import fractional_matrix_power as fmp
from sklearn.metrics import *
import scipy.linalg as sl
import scipy 
import time
from .plotting_functions import master_plotter


def fourier_mode_2d(freqs_Hz, times_sec, modes, box=None):
    """
    Construct a set of 2D Fourier modes from a list of wavenumber integers, 
    to form an incomplete set of 2D Fourier modes.

    Parameters
    ----------
    freqs_Hz (array_like):
        Frequency array, in Hz. Should be ordered.
        
    times_sec (array_like):
        Time array, in hours. Should be ordered.

    modes (list of tuple of int):
        List of mode integer pairs to include in operator.

    box (tuple of tuple):
        NOT IMPLEMENTED
        Keep all modes within a box, defined by the tuple:
        `((delay_min, delay_max), (frate_min, frate_max))`.
        The delays are in ns and the fringe rates in mHz.
    """
    Nfreqs, Ntimes = freqs_Hz.size, times_sec.size
    
    # Get grid spacing in expected units
    dfreq = (freqs_Hz[1] - freqs_Hz[0])
    dtime = (times_sec[1] - times_sec[0])

    # Get FFT wavenumbers
    kfreq = np.fft.fftfreq(Nfreqs, d=dfreq) # sec #* 1e9 # ns
    ktime = np.fft.fftfreq(Ntimes, d=dtime) # Hz * 1e3 # mHz

    # Get FFT mode integers
    nfreq = (np.fft.fftfreq(Nfreqs) * Nfreqs).astype(int)
    ntime = (np.fft.fftfreq(Ntimes) * Ntimes).astype(int)

    # Frequency/time grids with respect to origin
    f = freqs_Hz - freqs_Hz[0]
    t = times_sec - times_sec[0]

    # Get indices of modes we want to keep
    basis_fns = np.zeros((len(modes), Nfreqs, Ntimes), dtype=np.complex128)
    for i, mode in enumerate(modes):
        nf, nt = mode
        print(nf, nt)
        assert isinstance(nf, int), "modes must only contain pairs of integers"
        assert isinstance(nt, int), "modes must only contain pairs of integers"
        assert nf in nfreq, "Delay mode nf=%d not in available range (%d -- %d)." \
            % (nf, nfreq.min(), nfreq.max())
        assert nt in ntime, "Fringe rate mode nt=%d not in available range (%d -- %d)." \
            % (nt, ntime.min(), ntime.max())

        # Get mode indices
        idx_f = np.where(nfreq == nf)[0][0]
        idx_t = np.where(ntime == nt)[0][0]
        #mode_idxs.append( (idx_f, idx_t) )

        print(kfreq[idx_f], ktime[idx_t])

        # Add basis function to operator
        basis_fns[i] = np.exp(2.*np.pi*1.j * (  kfreq[idx_f] * f[:,np.newaxis]
                                     + ktime[idx_t] * t[np.newaxis,:] ) ) \
                     / np.sqrt(Nfreqs * Ntimes)
        
    return basis_fns, kfreq * 1e9, ktime * 1e3


def sys_modes(freqs_Hz, times_sec, modes):
    """
    Construct systematic mode operator, which is a 2D Fourier basis.
    """
    u, kfreq, ktime = fourier_mode_2d(freqs_Hz=freqs_Hz, 
                                      times_sec=times_sec, 
                                      modes=modes)
    return u.reshape((u.shape[0], -1)).T


def sq_mat_tr(A,flag='r'):
    '''
    Convert A_mat from the GCR equation into a square matrix for linear system solver.(Method 1)
    Original matrix A has shape of (2,2,n,n). This returns a matrix with shape 2*n,2*n

    Parameters:
        A: Matrix of shape (2,2,n,n)

    Returns:
        reshaped_A: Matrix A reshaped to (2*n,2*n)
    '''
    sh=np.shape(A)
    if flag=='c':
        reshaped_A = np.array(A).transpose(0, 3, 1, 2).reshape(sh[0] * sh[2], sh[1] * sh[3])
    elif flag=='r':
        reshaped_A = np.array(A).transpose(0, 2, 1, 3).reshape(sh[0] * sh[2], sh[1] * sh[3])
    return reshaped_A

def sq_mat_tr2(your_mat):
    '''
    Convert A_mat from the GCR equation into a square matrix for linear system solver.(Method 2)
    Original matrix your_mat has shape of (2,2,n,n). This returns a matrix with shape 2*n,2*n
    
    Parameters:
        A: Matrix of shape (2,2,n,n)
        
    Returns:
        reshaped_A: Matrix A reshaped to (2*n,2*n)
    '''
    sh=np.shape(your_mat)
    your_mat=np.array(your_mat)
    N=sh[2]
    square_mat = np.zeros((2*N, 2*N), dtype=your_mat.dtype) # empty matrix of the right type

    square_mat[:N,:N] = your_mat[0,0,:,:]
    square_mat[:N,N:] = your_mat[0,1,:,:]
    square_mat[N:,:N] = your_mat[1,0,:,:]
    square_mat[N:,N:] = your_mat[1,1,:,:]
    return square_mat

def inv_mat(mat):
    '''
    Inverses an invertible diagonal matrix mat without using np.linalg.inv()
    
    Paramters:
        mat: invertible matrix
    
    Returns:
        mat_inv: mat inverted
    '''
    mat=np.array(mat)
    diag_el=np.diag(mat)
    diag_inv=1/diag_el
    mat_inv=np.zeros(np.shape(mat))
    np.fill_diagonal(mat_inv,diag_inv)
    return mat_inv

def cholesky_inverse(A):
    """
    Inverts a positive-definite matrix A using Cholesky decomposition.
    
    Args:
    - A: A positive-definite matrix
    
    Returns:
    - A_inv: The inverse of matrix A
    """
    # Ensure A is a NumPy array
    A = np.array(A)
    
    # Perform Cholesky decomposition
    L = np.linalg.cholesky(A)
    
    # Solve L * L.T = A for A^-1 using forward and backward substitutions
    
    # Step 1: Solve L * y = I for y using forward substitution
    n = A.shape[0]
    I = np.eye(n)
    y = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            temp_sum = sum(L[i, k] * y[k, j] for k in range(i))
            y[i, j] = (I[i, j] - temp_sum) / L[i, i]
    
    # Step 2: Solve L.T * x = y for x using backward substitution
    L_T = L.T
    A_inv = np.zeros_like(A)
    for i in range(n-1, -1, -1):
        for j in range(n):
            temp_sum = sum(L_T[i, k] * A_inv[k, j] for k in range(i+1, n))
            A_inv[i, j] = (y[i, j] - temp_sum) / L_T[i, i]
    
    return A_inv

def gcr_sys_v1(Binv,d,Ninv,s,H, b_sys_past=None, verbose=False,iter=0):
    '''
    Parameters:
        Binv: array_like
            Inverse of Systematics covariance (Nmodes,Nmodes)
        d: array_like
            data (visbilities) shape (Nfreqs,Ntimes)
        Ninv: array_like
            Inverse of noise covariance matrix
        s: array_like
            sky model (Nfreqs,Ntimes)
        H: array_like
            Systematics basis functions with shape (Nfreqs*Ntimes,Nmodes)
        b_sys_past: array_like
            Last estimate of the systematics coefficients (Nmodes,)
        verbose: Bool
            Verbosity of printing results
        iter: int
            Iteration of Gibbs sampler for plotting
        
    Returns:
        b_sys: array_like
            Vector of systematics coefficients (Nmodes,)
    '''
    if verbose:
        st=time.time()
    Ntimes, Nfreqs= d.shape
    Nmodes = H.shape[1]
    master_plotter([d],col_labels=[' '],fig_title='Data residual sent to gcr_sys iter'+str(iter))  #Plotting data sent into the solver    
    #Flattening datasets for operation
    d=d.flatten(order='F')
    s=s.flatten(order='F')
    # master_plotter([s.reshape((Ntimes,Nfreqs),order='F')],col_labels=[' '],fig_title='Sky model sent to gcr_sys iter'+str(iter)) #Plotting the sky model     
    # master_plotter([Binv],col_labels=[' '],fig_title='B inverse',imag_flag=False) #Plotting the Cov matrix
    
    Binv_diag=np.diag(Binv)
    Binv_exp=np.concatenate((Binv_diag.real,Binv_diag.real))*np.eye(2*np.shape(Binv)[0]) #Explanded Binv for the realified case [[Binv,0],[0,Binv]]
    
    # master_plotter([Binv_exp],col_labels=[' '],fig_title='Binv realified',imag_flag=False) #Plotting the expanded Binv
    
    
    diag_el=Ninv[0,0]
    Ninv=diag_el*np.ones(shape=Ntimes*Nfreqs, dtype=complex)
    Nih=np.sqrt(Ninv)

    #Complex Gaussian vectors with unit variance for fluctuations     
    om_re=np.random.normal(size=(Nfreqs*Ntimes),scale=1/np.sqrt(2),loc=0) #Real part
    om_im=np.random.normal(size=(Nfreqs*Ntimes),scale=1/np.sqrt(2),loc=0) #Imaginary part
    
    
    '''eq A'''
    Nih_sre=Nih*s.real #N^-1/2 * s.real
    Nih_sim=Nih*s.imag #N^-1/2 * s.imag
    
    # master_plotter([Nih_sre.reshape((Ntimes,Nfreqs),order='F'),Nih_sim.reshape((Ntimes,Nfreqs),order='F')],col_labels=['sqrt(Ninv)*s.real*omega.real','sqrt(Ninv)*s.imag*omega.imag'],fig_title='Nih*sre*omre comparison')
    
    #Making the M_tilde sub-matrix
    m11= Nih_sre[:,np.newaxis]*H.real - Nih_sim[:,np.newaxis]*H.imag
    m12= -1 *Nih_sre[:,np.newaxis]*H.imag - Nih_sim[:,np.newaxis]*H.real
    
    # master_plotter([m11,m12],col_labels=['M11 element','M12 element'],fig_title='M_tile element comparison')

    nume=np.concatenate((m11,m12),axis=1) #Numerator of M_tilde
    denom=np.concatenate((-1*m12,m11),axis=1) #Denominator of M_tilde
    M_tilde=np.concatenate((nume,denom),axis=0) 
    
    # master_plotter([nume,denom],col_labels=['Real','Imaginary'],fig_title='M_tilde matrix (realified)',plot_type='matshow',imag_flag=False)

    #Putting A matrix together
    A_mat= Binv_exp + M_tilde.conj().T @ M_tilde # Try einsum as an alternative
    
    # master_plotter([A_mat[:Nmodes,:],A_mat[Nmodes:,:]],col_labels=['Real','Imaginary'],fig_title='A matrix',plot_type='matshow',imag_flag=False)

    nih_dre=Nih*d.real #N^-1/2 * d.real
    nih_dim=Nih*d.imag #N^-1/2 * d.imag

    #Multiplying gaussian fluctuations
    Nih_sre = Nih_sre*om_re  
    Nih_sim = Nih_sim*om_im
    # master_plotter([nih_dre.reshape((Ntimes,Nfreqs),order='F'),nih_dim.reshape((Ntimes,Nfreqs),order='F')],col_labels=['sqrt(Ninv)*d.real','sqrt(Ninv)*d.imag'],fig_title='sqrt(Ninv)*data comparison',imag_flag=False)

    nume= m11.T @ nih_dre + -1 * m12.T @nih_dim + (H.real.T @ Nih_sre) + (H.imag.T @ Nih_sim) #Numerator of b_mat
    denom= m12.T @ nih_dre + m11.T @ nih_dim - (H.imag.T @ Nih_sre) + (H.real.T @ Nih_sim) #Denominator of b_mat
    
    b_mat= np.concatenate((nume, denom), axis=0)
    
    Ai = np.linalg.inv(A_mat) #Pseudo-inverse for preconditioning
    
    # master_plotter([Ai],col_labels=[' '],fig_title='Pseudo inverse of A matrix',plot_type='matshow',imag_flag=False)
    
    if b_sys_past is not None:
        x0=np.concatenate([b_sys_past.real,b_sys_past.imag],axis=0)

    b_sys,info=scipy.sparse.linalg.cgs(A_mat,b_mat, M=Ai, tol=1e-12)
    
    residuals = np.abs(A_mat @ b_sys - b_mat).mean()
    b_sys=b_sys[:int(len(b_sys)/2)] + 1.j* b_sys[int(len(b_sys)/2):]  #Separating the real and imaginary components

    if verbose:
        print(f"{time.time() - st:<12.1f}", end="\t")
        print(f"{info:<8.1f}", end=" ")
        print(f"{residuals:<12.2e}", end="")
    
    return b_sys