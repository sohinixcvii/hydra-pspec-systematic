# Function definitions for GCR solver

import numpy as np
from math import pi as pi
import matplotlib.pylab as plt
from scipy.linalg import fractional_matrix_power as fmp
from sklearn.metrics import *
import scipy.linalg as sl
import scipy 
import time

def fourier_2d(freqs,times):
    F, T = np.meshgrid(freqs, times)

    n_freq = len(freqs)
    n_time = len(times)
    delta_f = freqs[1] - freqs[0]
    delta_t = times[1] - times[0]

    # Creating the delay and fringe_rate axes
    delay = np.fft.fftfreq(n_freq, delta_f)
    fringe_rate = np.fft.fftfreq(n_time, delta_t)

    # Creating the Fourier space grid
    DELAY, FR = np.meshgrid(delay, fringe_rate)
    return DELAY,FR

def fourier_mode_2d_udf(freqs, times, nfreq, ntime, freq0=None, time0=None, shape0=None):
    
    """
    Construct a set of 2D Fourier modes from a list of wavenumber integers, 
    to form an incomplete set of 2D Fourier modes.
    """
    # print("Modes: {}".format([ntime,nfreq]))
    freqs=freqs*1e-9

    # Decide on origin of frequency axis for FT
    if time0 is None:
        time0 = times[0]
    if freq0 is None:
        freq0 = freqs[0]

    # Determine normalising factors. If being used as a standalone Fourier operator, 
    # these are just the lengths of the freq and time arrays. If being used as a 
    # chunk of a Fourier operator across multiple workers, use the overall shape 
    # from 'shape0'
    if shape0 is None:
        Nfreqs = freqs.size
        Ntimes = times.size
    else:
        Nfreqs, Ntimes = shape0

    # Build grid of freqs and times
    nfreq = np.atleast_1d(nfreq)
    ntime = np.atleast_1d(ntime)
    assert len(nfreq.shape) == 1
    assert len(ntime.shape) == 1
    assert len(freqs.shape) == 1
    assert len(times.shape) == 1
    t2d, f2d = np.meshgrid(times - time0, freqs - freq0)

    # Calculate wavenumbers for each mode
    kfreq =(2 * np.pi * nfreq)
    ktime = (2 * np.pi * ntime)

    # Shape: (Nmodes, Nfreqs, Ntimes)
    basis_fns = np.exp(1.j \
                        * (  (kfreq[:,np.newaxis,np.newaxis] * f2d[np.newaxis,:,:]) \
                           + (ktime[:,np.newaxis,np.newaxis] * t2d[np.newaxis,:,:])) ) \
              / np.sqrt(Nfreqs * Ntimes)
    return basis_fns

#Function defining the U_sys operator
def h_j_op(freqs,lsts,nm_list):
    '''loop through the nm_list'''
    u=np.array([fourier_mode_2d_udf(freqs,lsts,n,m).flatten() for n,m in nm_list])
    return u.T

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

def gcr_sys_v1(Binv,d,Ninv,s,H, b_sys_past=None, verbose=False):
    '''
    Parameters:
    Binv:

    d:
        data (visbilities) flattened to a vector of shape (Nfreqs*Ntimes,)
    N:
        Noise covariance simplified to a vector with shape (Nfreqs*Ntimes,)
    s:
        sky model flattened to a vector of shape (Nfreqs*Ntimes,)
    H:
        Systematics basis functions with shape (Nfreqs*Ntimes,Nmodes)
    '''
    # d_exp=np.concatenate((d.real,d.imag)) #.reshape([-1,1])
    # Bi_diag=0.01
    if verbose:
        st=time.time()
    Ntimes, Nfreqs= d.shape
    d=d.flatten(order='F')
    Binv_diag=np.diag(Binv)
    Binv_exp=np.concatenate((Binv_diag.real,Binv_diag.imag))*np.eye(2*np.shape(Binv)[0])
    diag_el=Ninv[0,0]
    Ninv=diag_el*np.ones(shape=Ntimes*Nfreqs, dtype=complex)  #.reshape([len(d),],order='F')
    Nih=np.sqrt(Ninv)
    # print("Shape of realified matrices: \n d: {},\n Binv: {}\n Nih: {}\n s.real: {}\n s.imag: {}".format(d.shape,Binv_exp.shape,Nih.shape,s.real.shape,s.imag.shape))
    
    '''eq A'''
    Nih_sre=Nih*s.real
    Nih_sim=Nih*s.imag
    
    # print("Shape check 2: \n H: {}\n Nih_sre: {}\n Nih_sim: {}".format(H.shape,Nih_sre.shape,Nih_sim.shape))
    
    m11= Nih_sre[:,np.newaxis]*H.real - Nih_sim[:,np.newaxis]*H.imag
    m12= -1 *Nih_sre[:,np.newaxis]*H.imag - Nih_sim[:,np.newaxis]*H.real
    # m21= Nih_sim[:,np.newaxis]*H.real + Nih_sre[:,np.newaxis]*H.imag
    # m22= -Nih_sim[:,np.newaxis]*H.imag + Nih_sre[:,np.newaxis]*H.real
    

    nume=np.concatenate((m11,m12),axis=1)
    denom=np.concatenate((-1*m12,m11),axis=1)
    M_tilde=np.concatenate((nume,denom),axis=0)
    # print("Component shape checks: \n m11: {}\n m12: {}\n M_tilde: {}".format(m11.shape,m12.shape,M_tilde.shape))    
    A_mat= Binv_exp + M_tilde.conj().T @ M_tilde # Try einsum as an alternative
    
    nih_dre=Nih*d.real
    nih_dim=Nih*d.imag
    
    om_re=np.random.normal(size=(Nfreqs),scale=1/np.sqrt(2),loc=0)
    om_im=np.random.normal(size=(Nfreqs),scale=1/np.sqrt(2),loc=0)
    
    #FIXME: add the gaussian fluctuations to the following terms
    nume= m11.T @ nih_dre + -1 * m12.T @nih_dim + (H.real.T @ Nih_sre) * om_re + (H.imag.T @ Nih_sim) * om_im
    denom= m12.T @ nih_dre + m11.T @ nih_dim - (H.imag.T @ Nih_sre) * om_re + (H.real.T @ Nih_sim) * om_im
    
    b_mat= np.concatenate((nume, denom), axis=0)
    
    # print("Shape checks: \n A_mat: {}\n b_mat: {}".format(A_mat.shape,b_mat.shape))
    Ai = np.linalg.inv(A_mat)
    if b_sys_past is not None:
        x0=np.concatenate([b_sys_past.real,b_sys_past.imag],axis=0)
    b_sys,info=scipy.sparse.linalg.cgs(A_mat,b_mat, M=Ai, tol=1e-12)
    residuals = np.abs(A_mat @ b_sys - b_mat).mean()
    b_sys=b_sys[:int(len(b_sys)/2)] + 1.j* b_sys[int(len(b_sys)/2):]
    # print("Sys time: ",f"{time.time() - st:<12.1f}")
    if verbose:
        print(f"{time.time() - st:<12.1f}", end="\t")
        print(f"{info:<8.1f}", end=" ")
        print(f"{residuals:<12.2e}", end="")
    
    return b_sys