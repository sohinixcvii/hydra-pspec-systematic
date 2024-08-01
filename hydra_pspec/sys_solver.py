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
    Lfreq = (freqs[1] - freqs[0]) * freqs.size
    Ltime = (times[1] - times[0]) * times.size

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
    # kfreq = (2. * np.pi * nfreq / Lfreq)*1e-9 # inverse freq. units
    # ktime = 2. * np.pi * ntime / Ltime # inverse time units
    # FIXME: new kfreq test
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

#Function creating synthetic data
def synth_data(coeff,comp,noise,w):  #coeff: (nxm), comp:(mx1)
    '''
    Synthesize data from operator, noise, component, and gaussian vector
    coeff: h_j operator, complex, shape: len(frequencies)*len(lsts),len(nm_list)
    comp: Complex systematic component vector, shape: len(nm_list)
    noise: Symmetric, square, real valued noise covariance matrix, shape: len(frequencies)*len(lsts),len(frequencies)*len(lsts)
    w: complex gaussian vector with 0 mean and unit variance, shape:len(frequencies)*len(lsts)
    '''
    d=coeff @ comp+fmp(noise,0.5) @ w
    return d

def sq_mat_tr(A,flag='r'):
    '''
    Convert A_mat from the GCR equation into a square matrix for linear system solver.(Method 1)
    Original matrix A has shape of (2,2,n,n). This returns a matrix with shape 2*n,2*n
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

def mat_solver(b_sys,noise_f,w,h_j,B):
    '''
    Implements the GCR equation. Forms matrices and solves the system of equations. 
    b_sys: Complex systematic component vector, shape: len(nm_list)
    noise_f: Symmetric, square, real valued noise covariance matrix, shape: len(frequencies)*len(lsts),len(frequencies)*len(lsts)
    w: complex gaussian vector with 0 mean and unit variance, shape:len(frequencies)*len(lsts)
    h_j: h_j projection operator, complex, shape: len(frequencies)*len(lsts),len(nm_list)
    B: Prior covariance matrix (must be convertible)
    '''
    data_s=synth_data(h_j,b_sys,noise_f,w)
    n_i=inv_mat(noise_f)
    B_i=inv_mat(B)
    # print("\n Synthetic data: ",data_s)
    a11=B_i+ np.real(h_j).T @ n_i @ np.real(h_j) + np.imag(h_j).T @ n_i @ np.imag(h_j)
    a12=np.imag(h_j).T @ n_i @ np.real(h_j) - np.real(h_j).T @ n_i @ np.imag(h_j)
    a21=np.real(h_j).T @ n_i @ np.imag(h_j) - np.imag(h_j).T @ n_i @ np.real(h_j)
    a22=B_i+ np.imag(h_j).T @ n_i @ np.imag(h_j) + np.real(h_j).T @ n_i @ np.real(h_j)

    b111=np.real(h_j).T @ n_i @ np.real(data_s)
    b112=np.imag(h_j).T @ n_i @ np.imag(data_s)
    b113=np.real(h_j).T @ fmp(noise_f,-0.5) @ np.real(w)
    b114=np.imag(h_j).T @ fmp(noise_f,-0.5) @ np.imag(w)
    b11=b111+b112+b113+b114

    b211=-1*np.imag(h_j).T @ n_i @ np.real(data_s)
    b212=np.real(h_j).T @ n_i @ np.imag(data_s)
    b213=-1*np.imag(h_j).T @ fmp(noise_f,-0.5) @ np.real(w)
    b214=np.real(h_j).T @ fmp(noise_f,-0.5) @ np.imag(w)

    b21=b211+b212+b213+b214

    A_mat=[[a11, a12],[a21, a22]]

    sqA=sq_mat_tr2(A_mat)

    b_mat=np.append(b11,b21)

    # b_pred=sl.solve(sqA,b_mat)
    b_pred,info=scipy.sparse.linalg.cg(sqA,b_mat,tol=1e-10)

    return b_pred, data_s, sqA, b_mat

def mat_solver_dum(data_s,noise_f,w,h_j,B):
    '''
    Implements the GCR equation. Forms matrices and solves the system of equations. 
    b_sys: Complex systematic component vector, shape: len(nm_list)
    noise_f: Symmetric, square, real valued noise covariance matrix, shape: len(frequencies)*len(lsts),len(frequencies)*len(lsts)
    w: complex gaussian vector with 0 mean and unit variance, shape:len(frequencies)*len(lsts)
    h_j: h_j projection operator, complex, shape: len(frequencies)*len(lsts),len(nm_list)
    B: Prior covariance matrix (must be convertible)
    '''
    n_i=inv_mat(noise_f)
    B_i=inv_mat(B)
    # print("\n Synthetic data: ",data_s)
    a11=B_i+ np.real(h_j).T @ n_i @ np.real(h_j) + np.imag(h_j).T @ n_i @ np.imag(h_j)
    a12=np.imag(h_j).T @ n_i @ np.real(h_j) - np.real(h_j).T @ n_i @ np.imag(h_j)
    a21=np.real(h_j).T @ n_i @ np.imag(h_j) - np.imag(h_j).T @ n_i @ np.real(h_j)
    a22=B_i+ np.imag(h_j).T @ n_i @ np.imag(h_j) + np.real(h_j).T @ n_i @ np.real(h_j)

    b111=np.real(h_j).T @ n_i @ np.real(data_s)
    b112=np.imag(h_j).T @ n_i @ np.imag(data_s)
    b113=np.real(h_j).T @ fmp(noise_f,-0.5) @ np.real(w)
    b114=np.imag(h_j).T @ fmp(noise_f,-0.5) @ np.imag(w)
    b11=b111+b112+b113+b114

    b211=-1*np.imag(h_j).T @ n_i @ np.real(data_s)
    b212=np.real(h_j).T @ n_i @ np.imag(data_s)
    b213=-1*np.imag(h_j).T @ fmp(noise_f,-0.5) @ np.real(w)
    b214=np.real(h_j).T @ fmp(noise_f,-0.5) @ np.imag(w)

    b21=b211+b212+b213+b214

    A_mat=[[a11, a12],[a21, a22]]

    sqA=sq_mat_tr2(A_mat)

    b_mat=np.append(b11,b21)

    # b_pred=sl.solve(sqA,b_mat)
    b_pred,info=scipy.sparse.linalg.cg(sqA,b_mat,tol=1e-10)

    return b_pred, data_s, sqA, b_mat


def gcr_sys(vis,Ninv,B,nm_list, times, freqs, h_j=None):
    '''
    Implements the GCR equation. Forms matrices and solves the system of equations to obtain the systematics vector b_sys. 
    Parameters:

    vis: Visibilities shape=(len(lsts)*len(freqs))
    Ninv: Inverse of noise covariance matrix shape=(len(frequencies)*len(LSTs),len(frequencies)*len(LSTs))
    w: complex gaussian vector with 0 mean and unit variance, shape=len(frequencies)*len(lsts)
    h_j: h_j projection operator, complex, shape=len(frequencies)*len(lsts),len(nm_list)
    B: Prior covariance matrix (must be invertible and diagonal) shape=(len(mode_pairs),len(mode_pairs))

    Returns:
    b_sys: Complex vector of systematic amplitude predictions of shape=`(len(nm_list))`.
    '''
    # t0=time.time()
    if h_j is None:
        h_j=h_j_op(freqs=freqs, lsts=times, nm_list=nm_list)
    
    B_i=inv_mat(B)

    vis_f=vis.flatten()
    # t1=time.time()
    w_re=np.random.normal(size=(len(vis_f),1),scale=1/np.sqrt(2),loc=0)
    w_im=np.random.normal(size=(len(vis_f),1),scale=1/np.sqrt(2),loc=0)
    w=w_re+w_im*1.j
    # t2=time.time()
    #FIXME: temporary solution for the issue with noise matrix size. Given noise matrix is of shape (freq,freq). We need a diag matrix of shape (flattened_data,flattened_data)
    diag_el=Ninv[0,0]
    Ninv=diag_el*np.ones(shape=np.shape(vis_f)[0], dtype=complex).reshape([np.shape(vis_f)[0],1])
    Ninv_sqrt=np.sqrt(Ninv)
    # t3=time.time()
    # print("\n \nNinv and ninv_sqrt vectors made in time: ",t3-t0)
    a11=B_i+ np.real(h_j).T @ (Ninv*np.real(h_j)) + np.imag(h_j).T @ (Ninv*np.imag(h_j))
    a12=np.imag(h_j).T @ (Ninv*np.real(h_j)) - np.real(h_j).T @ (Ninv*np.imag(h_j))
    a21=np.real(h_j).T @ (Ninv*np.imag(h_j)) - np.imag(h_j).T @ (Ninv*np.real(h_j))
    a22=B_i+ np.imag(h_j).T @ (Ninv*np.imag(h_j)) + np.real(h_j).T @ (Ninv*np.real(h_j))
    # t4=time.time()

    # print("A mat elements made in time: ",t4-t3)

    b111=np.real(h_j).T @ (Ninv.T*np.real(vis_f)).T[:,0]
    b112=np.imag(h_j).T @ (Ninv.T*np.imag(vis_f)).T[:,0]
    b113=(np.real(h_j).T @ (Ninv_sqrt*np.real(w)))[:,0]
    b114=(np.imag(h_j).T @ (Ninv_sqrt*np.imag(w)))[:,0]
    b11=b111+b112+b113+b114
    # t5=time.time()

    # print("Row 1 of b mat made in time: ",t5-t4)

    b211=-1*np.imag(h_j).T @ (Ninv.T*np.real(vis_f)).T[:,0]
    b212=np.real(h_j).T @ (Ninv.T*np.imag(vis_f)).T[:,0]
    b213=(-1*np.imag(h_j).T @ (Ninv_sqrt*np.real(w)))[:,0]
    b214=(np.real(h_j).T @ (Ninv_sqrt*np.imag(w)))[:,0]
    b21=b211+b212+b213+b214
    t6=time.time()

    # print("Row 2 of b mat made in time: ",t6-t5)
    A_mat=[[a11, a12],[a21, a22]]
    
    sqA=sq_mat_tr2(A_mat)
    
    b_mat=np.append(b11,b21)
    # t7=time.time()
    # print("A mat and b mat collated, a mat turned into square mat in time: ", t7-t6)
    # b_pred=sl.solve(sqA,b_mat)
    b_sys,_=scipy.sparse.linalg.cg(sqA,b_mat,tol=1e-10)
    # t8=time.time()

    # print("Solver ran in time: ",t8-t7)

    n_half=b_sys.shape[0]//2
    b_real=b_sys[:n_half]
    b_imag=b_sys[n_half:]
    b_sys=b_real+1.j*b_imag
    # t9=time.time()

    # print("Solution turned into complex vector in time: ",t9-t8)
    # print("Total func run time: ",t9-t0,"\n")
    return b_sys