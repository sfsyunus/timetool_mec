#!/usr/bin/env python
# -*- coding: utf-8 -*-
# to run on 8 cores:
# > mpiexec -n 8 python timetool.py 

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

#params = {'text.usetex': 'true', 'font.size': 20, 'legend.fontsize': 15, 'figure.figsize': [17.5, 11]}
params = {'font.size': 20, 'legend.fontsize': 15, 'figure.figsize': [17.5, 11]}
mpl.rcParams.update(params)

start_time = datetime.datetime.now()
# enables auto completion of hdf5 file content: very useful
#h5py.enable_ipython_completer()

# define the error function used to fit the edge and get the centroid
def error_function(x, a, b, c, d):
   return d + (a - d) / (1. + (x / c)**b)

def gaussian(x, m, s, a):
   return (a / (s * np.sqrt(2. * np.pi))) * (np.exp(-(x - m)**2/(2. * s**2)))


#plt.ion()
rank = MPI.COMM_WORLD.rank
#path = '/Users/egaltier/Data/Campagnes/2016/LCLS/timetool/data/meclt9417/'
path = '/reg/d/psdm/mec/meclt9417/hdf5/'
f = h5py.File(path + 'meclt9417-r0410.h5', 'r', driver='mpio', comm=MPI.COMM_WORLD)
bg = h5py.File(path + 'meclt9417-r0409.h5', 'r', driver='mpio', comm=MPI.COMM_WORLD)

print('--> saving background and file data in object')
bg1 = bg['Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/MecTargetChamber.0:Opal4000.3/image']
#f1 = f['Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/MecTargetChamber.0:Opal4000.3/image']
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

# plot the background
#plt.imshow(bg_avg, interpolation='nearest', aspect='auto', cmap='plasma', clim=(0, 700))
#plt.colorbar()
#plt.show()
# plot the data
#plt.imshow(f1[1], interpolation='nearest', aspect='auto', cmap='plasma', clim=(0, 700))
#plt.colorbar()
#plt.show()

# remove zeros from the data file
f1_nz = np.where(f1[:]>0, f1[:], 1)
# divide bg by the various images
#res = bg1[6].astype(float)/f1_nz[0:1].astype(float)
#res = bg_avg/f1_nz.astype(float)
res = bg_avg-f1_nz.astype(float)
#print('res shape {}'.format(res.shape))
# get rid of the inf or NaN
res[~np.isfinite(res)] = 0
# create the matrix containing the rotated images
# res_rot=np.zeros([f1.shape[0], res.shape[1], res.shape[2]])
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
#proj = res_rot_crop.sum(axis=2)
#proj = res_rot_crop[:, :, 200:400].sum(axis=2)
proj = res_rot_crop[:, :, :].sum(axis=2)

# plot the projection of the rotated cropped image
f1_ax5 = fig.add_subplot(gs[2, 0])
#for i in xrange(1):
#   plt.plot(proj[i]/proj[i].max())

print('--> fitting the projections')
#min_fit = 550
min_fit = 0
# define x array: should be kept as 0 since it's to get an array lenght only
xd = np.arange(min_fit, min_fit+len(proj[0][min_fit:]))
edge = np.zeros(evt_data)

# plot the data
plt.plot(xd, proj[show][min_fit:])

# execute the fit on the right part of the data
for i in xrange(evt_data):
# for -200 fs
#    popt, pcov = curve_fit(error_function, xd, proj[i][min_fit:], bounds=([-1500, -20, 0, 0], [0, 0, 1000, 100000]))
# for -600 fs
#    popt, pcov = curve_fit(error_function, xd, proj[i][min_fit:], bounds=([0, 0, 0, -500], [20000, 10, 1000, 0]))
    popt, pcov = curve_fit(error_function, xd, proj[i][min_fit:])#, bounds=([0, 0, 0, -10000], [5, 10, 1000, 0]))
# for +600 fs
#    popt, pcov = curve_fit(error_function, xd, proj[i][min_fit:], bounds=([0, 0, 0, -2000], [15000, 100, 2000, 0]))
    print(popt)
    edge[i] = popt[2]
    print(i, edge[i])
# plot the fit and the center extracted from the fit a t a 'show' position
    if i == show:
        plt.plot(xd, error_function(xd, popt[0], popt[1], popt[2], popt[3]))
        plt.axvline(x = popt[2])


## defines the fitting function with a constrained variable for the first parameter
#def error_function_mean(x, a, b, c):
#   return c + (val - c) / (1. + (x / b)**a)
## get the index of the max value of the array
#mean=np.argmax(proj[4])
## get the mean value of the array around mean index: can be tuned
#val=np.mean(proj[4][mean-10:mean])
## fit the curve to the data using data from the mean to +200 to get the edge + some pedestal: can be tuned
#popt, pcov = curve_fit(error_function_mean, xd[mean:mean+200], proj[2][mean:mean+200])
## plot data and fit
#plt.plot(xd[9:], proj[4][9:])
#plt.plot(xd, error_function_mean(xd, popt[0], popt[1], popt[2]))

# plot the histogram of the edge position
print('--> plotting the histogram of the edge position')
#edge=r221*2.63
#edge=edge*2.63



# normalise to get the gaussian properly fitting with the sum under the guassian = 1
f1_ax6 = fig.add_subplot(gs[2, 1])
n, bins, patches = plt.hist(edge, bins='auto', density=True, align='left', facecolor='orange', alpha=0.7)
xi = np.linspace(np.mean(edge)-200, np.mean(edge)+200, 200)
#yi = np.interp(xi, bins[:len(bins)-1], n)

#xi = np.arange(min_fit-np.amax(edge), min_fit+np.amax(edge))
#xi = np.arange(np.amax(edge))
#yi = np.interp(xi, bins[:len(bins)-1], n)
#popt, pcov = curve_fit(gaussian, xi, yi)#, p0=(np.mean(edge), np.std(edge), np.max(edge)))

#(mu, sigma) = norm.fit(edge)
#y = mlab.normpdf(bins, mu, sigma)
#l = plt.plot(bins, y, 'r--', linewidth=2)

# lets try the normal distribution first
mean, std = stats.norm.fit(edge) # get mean and standard deviation  
print(mean, std)
pdf_g = stats.norm.pdf(xi, mean, std) # now get theoretical values in our interval  
plt.axvline(x = mean)
plt.plot(xi, pdf_g, label = '$\mu$={:2f} px\n $\sigma$={:2f} px'.format(mean, std)) # plot it
plt.legend()



#plt.plot(bins[:len(bins)-1], n)
#plt.plot(xi, gaussian(xi, popt[0], popt[1], popt[2]), color='red')

plt.show()

f.close()
bg.close()
# display time needed to execute the program
print("--- {} seconds ---".format(datetime.datetime.now() - start_time))

