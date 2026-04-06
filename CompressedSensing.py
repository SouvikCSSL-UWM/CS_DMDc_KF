# Author: Souvik Roy, Roshan M. D'souza
# Compressed sensing of image with OWL-QN Algorithm

from pylbfgs import owlqn
import numpy as np
import cv2 as cv
import scipy.fftpack as spfft
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Original Image Path
Path = "/Users/souvikroy/Desktop/PythonProjects/Temporature_history.csv"

# Defination of 2D DCT and IDCT
def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


nx = 101
ny = 101
Error =0.1
k =20
# while Error >1:
# create random sampling index vector
sample_sizes = 0.1 # % measurements
k = round(nx * ny * sample_sizes)
ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices

# for each sample size
Xorig = pd.read_csv(Path, header= None)
Xorig = Xorig.to_numpy()
# Xorig = Xorig[:,:100]
print(Xorig.shape)
normalized_data = (Xorig - Xorig.min().min()) / (Xorig.max().max() - Xorig.min().min())
ngrid, ntime = Xorig.shape
# ntime = 100

# Create the 10x10 pixel image
# plt.imshow(Xorig, cmap='jet', interpolation='none')
# plt.colorbar()
# plt.axis('off')  # Hide the axes
# plt.show()

Z = np.zeros([nx,ny], dtype='float')
masks = np.zeros([nx,ny], dtype='float')
# for each color channel
error = np.zeros([ntime,1])
reconstructed_state =np.zeros(Xorig.shape)
# for i in range(ntime):
for i in range(ntime):
    def evaluate(x, g, step):

        # we want to return two things: 
        # (1) the norm squared of the residuals, sum((Ax-b).^2), and
        # (2) the gradient 2*A'(Ax-b)

        # expand x columns-first
        x2 = x.reshape((nx, ny)).T

        # Ax is just the inverse 2D dct of x2
        Ax2 = idct2(x2)

        # stack columns and extract samples
        Ax = Ax2.T.flat[ri].reshape(b.shape)

        # calculate the residual Ax-b and its 2-norm squared
        Axb = Ax - b
        fx = np.sum(np.power(Axb, 2))

        # project residual vector (k x 1) onto blank image (ny x nx)
        Axb2 = np.zeros(x2.shape)
        Axb2.T.flat[ri] = Axb # fill columns-first

        # A'(Ax-b) is just the 2D dct of Axb2
        AtAxb2 = 2 * dct2(Axb2)
        AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

        # copy over the gradient vector
        np.copyto(g, AtAxb)

        return fx

    # nx = 17
    # ny = 17

    X = Xorig[:,i].reshape(nx,ny)
    sigma = 0.1  # Standard deviation of noise
    noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)

    X_noisy = X + noise
    # create images of mask (for visualization)
    Xm = np.ones(X.shape)
    Xm.T.flat[ri] = X_noisy.T.flat[ri]
    masks = Xm

    # take random samples of image, store them in a vector b
    b = X.T.flat[ri].astype(float)
    #b = np.expand_dims(b, axis=1)

    # perform the L1 minimization in memory
    Xat2 = owlqn(nx*ny, evaluate, None, 1)

    # transform the output back into the spatial domain
    Xat = Xat2.reshape(nx, ny).T # stack columns
    Xa = idct2(Xat)
    Z = Xa.astype('float')
    reconstructed_state[:,i] = Z.reshape(nx*ny)

    error[i] = np.sqrt(np.mean((Z-X)**2))
    print('error\n', error)

Error = np.sqrt(np.mean((reconstructed_state-Xorig)**2))
#print(reconstructed_state)

print(Error)

with open('Reconstructed_State.csv', 'w+') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ',')
    my_writer.writerows(reconstructed_state)

test_index = 995

# --- Side-by-side comparison: Ground Truth vs Reconstructed ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Ground Truth
im1 = axes[0].contourf(Xorig[:, test_index].reshape([nx, ny]), 
                       cmap='jet', vmin=0, vmax=100)
axes[0].set_title(f'Ground Truth State at t={test_index}', fontsize=12, weight='bold')
axes[0].axis('off')
cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
cbar1.ax.tick_params(labelsize=8)

# Reconstructed
im2 = axes[1].contourf(reconstructed_state[:, test_index].reshape([nx, ny]), 
                       cmap='jet', vmin=0, vmax=100)
axes[1].set_title(f'Compressed Sensed State at t={test_index}', fontsize=12, weight='bold')
axes[1].axis('off')
cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
cbar2.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.show()


# --- Scatter Plot Comparison ---
plt.figure(figsize=(2, 2))
plt.scatter(Xorig[:, test_index], reconstructed_state[:, test_index], 
            alpha=0.6, edgecolors='k', s=40, cmap='jet')
plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Ideal Match')
plt.xlabel('Ground Truth', fontsize=11, weight='bold')
plt.ylabel('Compressed Sensed', fontsize=11, weight='bold')
plt.title('Scatter Comparison', fontsize=12, weight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# --- Total Relative Error ---
TRE = np.sum(100 * np.abs(reconstructed_state[:, test_index] - Xorig[:, test_index]) 
             / np.sum(Xorig[:, test_index]))
print(f'Total Relative Error (TRE): {TRE:.3f}%')


# --- Noisy Image ---
plt.figure(figsize=(6, 5))
im = plt.imshow(X_noisy, cmap='jet', interpolation='nearest')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.axis('off')
plt.title('Noisy Image', fontsize=12, weight='bold')
plt.tight_layout()
plt.show()
