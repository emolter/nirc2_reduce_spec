#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from nirc2_reduce import image

class Dark:
    
    def __init__(self, fnames):
        self.dummy_fits = image.Image(fnames[0])
        imlist = []
        for f in fnames:
            imlist.append(image.Image(f).data)
        self.dark = np.median(np.asarray(imlist), axis = 0)
        
    def write(self, outfile):
        hdulist_out = self.dummy_fits.hdulist
        hdulist_out[0].header['OBJECT'] = 'DARK'
        hdulist_out[0].data = self.dark
        hdulist_out[0].writeto(outfile, overwrite=True)        