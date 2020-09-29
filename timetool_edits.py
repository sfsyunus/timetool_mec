#!/usr/bin/env python
import sys
import os
import datetime
import h5py
import tables
import numpy as np
import matplotlib as mpl

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.mlab as mlab
import matplotlib.patches as ptc

from mpi4py import MPI
from scipy import ndimage
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

start_time = datetime.datetime.now()

# define the error function used to fit the edge and get the centroid
def error_function(x, a, b, c, d):
   return d + (a - d) / (1. + (x / c)**b)

def gaussian(x, m, s, a):
   return (a / (s * np.sqrt(2. * np.pi))) * (np.exp(-(x - m)**2/(2. * s**2)))



path = '/reg/d/psdm/mec/meclt9417/hdf5/'
f = h5py.File(path + 'meclt9417-r0410.h5', 'r', driver='mpio', comm=MPI.COMM_WORLD)
bg = h5py.File(path + 'meclt9417-r0409.h5', 'r', driver='mpio', comm=MPI.COMM_WORLD)

print('--> saving background and file data in object')
bg1 = bg['Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/MecTargetChamber.0:Opal4000.3/image']
f2 = f['Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/MecTargetChamber.0:Opal4000.3/image']

f1 = f2[0:10]

show = 1

# get the first image of the array to plot latter for debugging
img = f1[show].reshape(f1[:].shape[1], f1[:].shape[2])

# prepare the figure environement
fig = plt.figure()
gs = gridspec.GridSpec(3, 2)

# plot the original image
f1_ax1 = fig.add_subplot(gs[0, 0])
plt.grid(True)
plt.imshow(img, interpolation='nearest', aspect='auto', cmap='plasma', clim=(0, 300))
plt.colorbar()

evt_bg = len(bg1)
evt_data = len(f1)
print 'Number of events for bg:', evt_bg
print 'Number of events for data:', evt_data
# get the average of all the backgrounds
bg_avg = bg1[:].mean(axis=0)

f1_ax2 = fig.add_subplot(gs[0, 1])
plt.grid(True)
plt.imshow(bg_avg, interpolation='nearest', aspect='auto', cmap='plasma', clim=(0, 300))
plt.colorbar()

# remove zeros from the data file
f1_nz = np.where(f1[:]>0, f1[:], 1)
# background subtraction
res = bg_avg-f1_nz.astype(float)
# get rid of the inf or NaN
res[~np.isfinite(res)] = 0
# create the matrix containing the rotated images
res_rot=np.zeros([evt_data, res.shape[1], res.shape[2]])
# rotate the images and store the result in a matrix 3D
print('--> rotating the images and store the result in a 3D matrix')
for i in xrange(evt_data):
   res_rot[i]=ndimage.rotate(res[i], -3.5, reshape=False)

# plot the rotated full image
f1_ax3 = fig.add_subplot(gs[1, 0])
img = res_rot[show].reshape(res_rot.shape[1], res_rot.shape[2])
plt.grid(True)
plt.imshow(img, interpolation='nearest', aspect='auto', cmap='plasma', clim=(0, 100))
plt.colorbar()

# crop the images: could be done earlier
print('--> cropping the images')
# for mecln0416
#res_rot_crop = res_rot[0:5, 0:1300, 736:1736]
# for meclt9417
res_rot_crop = res_rot[:, 550:1300, 1000:1440]

# plot the rotated cropped image
f1_ax4 = fig.add_subplot(gs[1, 1])
img = res_rot_crop[show].reshape(res_rot_crop.shape[1], res_rot_crop.shape[2])
plt.imshow(img, interpolation='nearest', aspect='auto', cmap='plasma', clim=(0, 100))
plt.colorbar()

# create the projection of the images along the right axis
print('--> creating the projection of the images along the right axis')
proj = res_rot_crop[:, :, :].sum(axis=2)

# plot the projection of the rotated cropped image
f1_ax5 = fig.add_subplot(gs[2, 0])

print('--> fitting the projections')
min_fit = 0
# define x array: should be kept as 0 since it's to get an array lenght only
xd = np.arange(min_fit, min_fit+len(proj[0][min_fit:]))
edge = np.zeros(evt_data)

# plot the data
plt.plot(xd, proj[show][min_fit:])

# execute the fit on the right part of the data
for i in xrange(evt_data):
    popt, pcov = curve_fit(error_function, xd, proj[i][min_fit:])#, bounds=([0, 0, 0, -10000], [5, 10, 1000, 0]))
    print(popt)
    edge[i] = popt[2]
    print(i, edge[i])
# plot the fit and the center extracted from the fit a t a 'show' position
    if i == show:
        plt.plot(xd, error_function(xd, popt[0], popt[1], popt[2], popt[3]))
        plt.axvline(x = popt[2])


# plot the histogram of the edge position
print('--> plotting the histogram of the edge position')

# normalise to get the gaussian properly fitting with the sum under the guassian = 1
f1_ax6 = fig.add_subplot(gs[2, 1])
n, bins, patches = plt.hist(edge, bins='auto', density=True, align='left', facecolor='orange', alpha=0.7)
xi = np.linspace(np.mean(edge)-200, np.mean(edge)+200, 200)

# lets try the normal distribution first
mean, std = stats.norm.fit(edge) # get mean and standard deviation  
print(mean, std)
pdf_g = stats.norm.pdf(xi, mean, std) # now get theoretical values in our interval  
plt.axvline(x = mean)
plt.plot(xi, pdf_g, label = '$\mu$={:2f} px\n $\sigma$={:2f} px'.format(mean, std)) # plot it
plt.legend()


plt.show()

f.close()
bg.close()
# display time needed to execute the program
print("--- {} seconds ---".format(datetime.datetime.now() - start_time))

