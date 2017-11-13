#!/usr/bin/env python
from nirc2_reduce import flats
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import boxcar


class specFlat(flats.Flats):
    
    def __init__(self, fnames_off,fnames_on):
        
        flats.Flats.__init__(self, fnames_off, fnames_on)
        
    def wl_response(self):
        '''Sum flat along spatial axis, smooth w/ 5pixel boxcar, 
        divide median averaged flat by this wavelength response curve'''

        curve = np.mean(self.flat, axis = 0)
        box = boxcar(5)/5.0
        curvesmooth = np.convolve(curve, box, mode = 'same')
        self.flat = self.flat / curvesmooth
        