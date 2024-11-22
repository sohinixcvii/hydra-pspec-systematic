# Function definitions for GCR solver

import numpy as np
# import skimage
# from skimage.io import imshow, imread
# from skimage.color import rgb2gray
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

def fourier_mode_2d_udf(freqs, times, nfreq, ntime, freq0=None, time0=None, 
                             shape0=None):
    
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

def gcr_sys(vis,s,Ninv,Bi,nm_list, times, freqs, hj=None):
    '''
    Implements the GCR equation. Forms matrices and solves the system of equations to obtain the systematics vector b_sys. 
    Parameters:

    Parameters:
        vis: Ratio of corrupted Visibilities and model shape=(len(lsts),len(freqs))
        s: Model for eor and foreground visibilities shape=(len(lsts),len(freqs))
        Ninv: Inverse of noise covariance matrix shape=(len(frequencies),len(frequencies))
        Bi: Inverse of prior covariance matrix shape=(len(mode_pairs),len(mode_pairs))
        nm_list: List of selected modes for analysis, shape= (len(nm_list),2)
        times: Array of lsts, shape= (Ntimes, 1)
        freqs: Array of frequencies, shape= (Nfreqs, 1)
        hj: hj projection operator, complex, shape=len(frequencies)*len(lsts),len(nm_list)

    Returns:
        b_sys: Complex vector of systematic amplitude predictions of shape=`(len(nm_list))`.
    '''
    # t0=time.time()
    if hj is None:
        hj=h_j_op(freqs=freqs, lsts=times, nm_list=nm_list)

    vis_f=vis.reshape([len(times)*len(freqs),1],order='F')
    s=s.flatten().reshape([len(times)*len(freqs),1],order='F')
    # t1=time.time()
    # print("data and model formatted. Time: {}".format(t1-t0))
    diag_el=Ninv[0,0]
    Ninv=diag_el*np.ones(shape=len(times)*len(freqs), dtype=complex).reshape([len(times)*len(freqs),1],order='F')
    Ninv_sqrt=np.sqrt(Ninv)
    # t2=time.time()
    # print("Noise cov and sqrt made. Time: {}".format(t2-t1))
    w_re=np.random.normal(size=(len(times)*len(freqs),1),scale=1/np.sqrt(2),loc=0)
    w_im=np.random.normal(size=(len(times)*len(freqs),1),scale=1/np.sqrt(2),loc=0)
    Bi_diag=np.concatenate((Bi.diagonal(),Bi.diagonal()))
    # Bi_diag=0.01
    Bi=Bi_diag*np.eye(2*len(nm_list))
    prod_1=s.real.T @ (Ninv*s.real) + s.imag.T @ (Ninv*s.imag)
    # t3=time.time()
    # print("Product made. Time:  {}".format(t3-t2))

    a11=hj.real.T @ (prod_1*hj.real) + hj.imag.T @ (prod_1 *hj.imag)
    a12= -1*hj.real.T @ (prod_1*hj.imag) + hj.imag.T @ (prod_1*hj.real)
    
    # t4=time.time()
    # print("Prod_1 calculated in time: {}".format(t4-t3))
    
    a_top=np.concatenate((a11,a12),axis=1)
    a_bottom=np.concatenate((-1*a12,a11),axis=1)
    A_mat=np.concatenate((a_top,a_bottom),axis=0)
    A_mat= Bi+A_mat

    # t5=time.time()
    # print("A_mat made in time: {}".format(t5-t4))
    prod_2_re=s.real.T @ Ninv
    prod_2_im=s.imag.T @ Ninv
    prod_3_re=s.real.T @ Ninv_sqrt
    prod_3_im=s.imag.T @ Ninv_sqrt
    b11= hj.real.T @ (prod_2_re*vis_f.real)
    b12= hj.imag.T @ (prod_2_im*vis_f.imag)
    b13= hj.real.T @ (prod_3_re*w_re)
    b14= hj.imag.T @ (prod_3_im*w_im)
    b111=b11+b12+b13+b14

    # t6=time.time()
    # print("B11 made: {}".format(t6-t5))

    b21= -1*hj.imag.T @ (prod_2_re*vis_f.real)
    b22= hj.real.T @ (prod_2_im*vis_f.imag)
    b23= -1*hj.imag.T @ (prod_3_re*w_re)
    b24= hj.real.T @ (prod_3_im*w_im)
    b222= b21+b22+b23+b24

    # t7=time.time()
    # print("B222 made: {}".format(t7-t6))

    b_mat=np.concatenate((b111,b222),axis=0)
    # t8=time.time()

    # print("b_mat made: {}".format(t8-t7))
    b_sys,_=scipy.sparse.linalg.cgs(A_mat,b_mat,rtol=1e-10)

    n_half=b_sys.shape[0]//2
    b_real=b_sys[:n_half]
    b_imag=b_sys[n_half:]
    b_sys=b_real+1.j*b_imag

    return b_sys