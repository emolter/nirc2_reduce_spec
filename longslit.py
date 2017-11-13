#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from nirc2_reduce import image, flats, filt
from scipy.signal import medfilt

class Longslit:
    
    def __init__(self, fname, wl_soln):
        '''wl_soln is file output by arc lamp wl calibration: 2-d array
        across detector where wl is at every point'''
        self.im = image.Image(fname)
        self.header = self.im.header
        self.data = self.im.data
        self.wl = image.Image(wl_soln).data
        
    def calibrate(self, flatf, badpxf, fl, biasf = None):
        '''flt is filter name, e.g. kp'''
        #dark
        if biasf != None:
            bias = image.Image(biasf).data
            self.data = self.data - bias

        #flat    
        flat = image.Image(flatf).data
        self.data = self.data / flat
        
        #bad pixel map
        badpx = image.Image(badpxf).data
        bad_indices = np.where(badpx == 0)
        smoothed = medfilt(self.data,kernel_size = 7)
        self.data[bad_indices] = smoothed[bad_indices]
        
        #bandpass shape
        bp_interp = filt.Filt(fl).interp_f
        bp_corr = bp_interp(self.wl)
        self.data = self.data / bp_corr
        self.data[np.where(bp_corr < 1.0e-2)] = np.nan
    
    def subtract_sky(self, skyf):
        sky = np.loadtxt(skyf).T
        plt.plot(sky[0], sky[1])
        plt.show()
        sky1, sky2 = np.meshgrid(sky[1], sky[1]) #just an easy way to make it 2-d
        self.data = self.data - sky1
    
    def plot(self):
        plt.imshow(self.data, origin = 'lower left')
        plt.show()
        
    def write(self, outfile):
        hdulist_out = self.im.hdulist
        hdulist_out[0].data = self.data
        hdulist_out[0].header['OBJECT'] = 'CAL_SPECTRUM'
        hdulist_out[0].writeto(outfile, overwrite=True)
            
    def plot_at_point(self, x):
        plt.plot(self.wl[x,:], self.data[x,:])
        plt.show()
        
    def plot_med(self, xmin, xmax):
        wls = self.wl[self.wl.shape[0]/2,:]
        fluxes = np.median(self.data[xmin:xmax,:], axis = 0)
        plt.plot(wls, fluxes)
        plt.show() 

    def save_at_point(self, outfile, x):
        wl = self.wl[x,:]
        flux = self.data[x,:]
        out = np.asarray([wl,flux]).T
        np.savetxt(outfile, out)
        
    def save_med(self, outfile, xmin, xmax):
        '''Make sky frame with this'''  
        wls = self.wl[self.wl.shape[0]/2,:]
        fluxes = np.median(self.data[xmin:xmax,:], axis = 0)
        out = np.asarray([wls,fluxes]).T
        np.savetxt(outfile, out)

        
        
        
        