#!/usr/bin/env python

from psana import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import sys
import os
import datetime

from scipy import ndimage
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import erf

start_time = datetime.datetime.now()

bg_ds = DataSource('exp=meclt9417:run=409')
data_ds = DataSource('exp=meclt9417:run=410')
tt_det = Detector('opal7_Time_Tool')

# Get average of background run
tt_bg = 0
for nevt, evt in enumerate(bg_ds.events()):
    img = tt_det.image(evt)
    tt_bg += img
bg_avg = tt_bg / (nevt+1)

# Plot background average
plt.imshow(bg_avg, clim=(0, 4*np.mean(bg_avg)))
plt.colorbar()
plt.show()

def errfunc(x, a, b, c, d):
    return d + (a - d) / (1. + (x / c)**b)

def gaussian(x, x0, sigma, a):
    return a*np.exp(-(x-x0)**2/(2.*sigma**2))

# Subtract background, rotate, crop data images, lineout, fit
edges = np.array([])
for nevt, evt in enumerate(data_ds.events()):
    if nevt == 10: break
    img = tt_det.image(evt)
    # Remove zeros
    img_nz = np.where(img>0, img, 1)
    # Do background subtraction
    img = img - bg_avg
    #img = bg_avg - img
    # Remove inf or NaN
    img[~np.isfinite(img)] = 0
    # Do rotation
    img = ndimage.rotate(img, -3.5, reshape=False)
    # Crop image
    img = img[550:1300, 1000:1440]
    # Plot rotated, cropped, bgd subtracted image
    plt.imshow(img, clim=(0, 4*np.mean(bg_avg)))
    plt.colorbar()
    plt.show()
    # Plot lineout of background subtracted image
    lineout = img.sum(axis=1)
    plt.plot(lineout)
    plt.show()
    
    # Execute the fit
    xdat = np.arange(len(lineout))
    popt, pcov = curve_fit(errfunc, xdat, lineout)
    edges = np.append(edges, popt[2])
    plt.plot(xdat, errfunc(xdat, popt[0], popt[1], popt[2], popt[3]))
    plt.plot(lineout)
    plt.axvline(x = popt[2])
    plt.show()

# normalise to get the gaussian properly fitting with the sum under the guassian = 1
n, bins, patches = plt.hist(edges, bins='auto', density=True, align='left', facecolor='orange', alpha=0.7)
xi = np.linspace(np.mean(edges)-200, np.mean(edges)+200, 200)

# lets try the normal distribution first
mean, std = stats.norm.fit(edges) # get mean and standard deviation
print(mean, std)
pdf_g = stats.norm.pdf(xi, mean, std) # now get theoretical values in our interval
plt.axvline(x = mean)
plt.plot(xi, pdf_g, label = '$\mu$={:2f} px\n $\sigma$={:2f} px'.format(mean, std)) # plot it
plt.legend()

plt.show()

# display time needed to execute the program
print("--- {} seconds needed to execute ---".format(datetime.datetime.now() - start_time))
