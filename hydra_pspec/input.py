import numpy as np

f0=50
f1=300
t0=500
t1=1500
sigma=0.2
mean=np.array([0,0])
# Grid dimensions
N = 50  # Number of points in each dimension
L_freq = 100  # Frequency range
L_time = 100  # Time range

# Bivariate Gaussian parameters in Fourier space
sigma_x = 0.0001
sigma_y = 0.0001
rho = 0.05  # Correlation coefficient
thresh_f=0.95