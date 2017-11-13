#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class phot:
    
    def __init__(self, fname):
        '''Just load a 2-column spectrum and a vega spectrum.'''
        spectrum = np.loadtxt(fname).T
        self.wl, self.cts = spectrum[0], spectrum[1]
        vega = np.loadtxt('/Users/emolter/Python/nirc2_reduce/filter_passbands/vega_spectrum.txt').T
        self.v_wl, self.v_f = vega[0]/10000, vega[1] * 10000 #from angstroms to microns. vega flux is now in erg s-1 cm-2 um-1

    def compute_conversion():
        '''Determine conversion between counts and flux at all wavelengths assuming
        the standard star has exactly the spectrum of Vega.
        return [wl (microns), flux_per_ct (erg s-1 cm-2 um-1 / ct) '''
        v_interp = interp1d(self.v_wl, self.v_f)
        flux_vega = v_interp(self.wl) #evaluate at wavelengths in star spectrum
        flux_per = flux_vega / self.cts #flux per count at each wl
        #what about the fact the spectrum is median averaged
        '''tbh not sure how to do this...'''
        self.flux_per = flux_per