#!/usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from nirc2_reduce import image
import matplotlib.cm as cm
from matplotlib.widgets import Cursor
import os
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel
from scipy.optimize import curve_fit
import scipy.signal
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import SmoothBivariateSpline
import warnings
from scipy.signal import medfilt


'''Classes and functions for finding wavelength solution using arc lamp.
Taking heavily from pydis'''

def _gaus(x, a, b, x0, sigma):
    """
    Simple Gaussian function, for internal use only

    Parameters
    ----------
    x : float or 1-d numpy array
        The data to evaluate the Gaussian over
    a : float
        the amplitude
    b : float
        the constant offset
    x0 : float
        the center of the Gaussian
    sigma : float
        the width of the Gaussian

    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b

def _CheckMono(wave):
    '''
    Check if the wavelength array is monotonically increasing. Return a
    warning if not. NOTE: because RED/BLUE wavelength direction is flipped
    it has to check both increasing and decreasing. It must satisfy one!

    Method adopted from here:
    http://stackoverflow.com/a/4983359/4842871
    '''

    # increasing
    up = all(x<y for x, y in zip(wave, wave[1:]))

    # decreasing
    dn = all(x>y for x, y in zip(wave, wave[1:]))

    if (up is False) and (dn is False):
        print("WARNING: Wavelength array is not monotonically increasing!")

    return

def line_trace(img, pcent, wcent, fmask=(1,), maxbend=10, display=False):
    '''
    Trace the lines of constant wavelength along the spatial dimension.

    To be run after peaks found in the HeNeAr lamp. Usually run internally
    to HeNeAr_fit()

    Method works by tracing up and down from the image center (slice) along
    each HeNeAr line by 1 pixel, fitting a gaussian to find the center.

    Parameters
    ----------
    img : 2d float
        the HeNeAr data
    pcent : float array
        the pixel center along the image slice of each HeNeAr line to trace
    wcent : float array
        the identified wavelength that corresponds to each peak's pixel center (pcent)
    fmask : float array, optional
        the illumination section to trace trace over in spatial dimension
    maxbend : int, optional
        How big of a width (in pixel units) to allow the bend in the HeNeAr
        line to search over (Default is 10). Probably doesn't need to be
        modified much.
    display : bool, optional
        should we display plot after? (Default is False)

    Returns
    -------
    xcent, ycent, wcent
    These are the arrays of X pixel (wavelength dimension), Y pixel
    (spatial dimension), and corresponding wavelengths of each HeNeAr line.
    '''
    xcent_big = []
    ycent_big = []
    wcent_big = []

    # the valid y-range of the chip
    if (len(fmask)>1):
        ydata = np.arange(img.shape[0])[fmask]
    else:
        ydata = np.arange(img.shape[0])

    ybuf = 10
    # split the chip in to 2 parts, above and below the center
    ydata1 = ydata[np.where((ydata>=img.shape[0]/2) &
                            (ydata<img.shape[0]-ybuf))]
    ydata2 = ydata[np.where((ydata<img.shape[0]/2) &
                            (ydata>ybuf))][::-1]

    # plt.figure()
    # plt.plot(img[img.shape[0]/2,:])
    # plt.scatter(pcent, pcent*0.+np.mean(img))
    # plt.show()

    img_med = np.nanmedian(img)
    # loop over every HeNeAr peak that had a good fit

    for i in range(len(pcent)):
        xline = np.arange(int(pcent[i])-maxbend,int(pcent[i])+maxbend)

        # above center line (where fit was done)
        for j in ydata1:
            yline = img[j-ybuf:j+ybuf, int(pcent[i])-maxbend:int(pcent[i])+maxbend].sum(axis=0)
            # fit gaussian, assume center at 0, width of 2
            if j==ydata1[0]:
                cguess = pcent[i] # xline[np.argmax(yline)]

            pguess = [np.nanmax(yline), img_med, cguess, 2.]
            try:
                popt,pcov = curve_fit(_gaus, xline, yline, p0=pguess)

                if popt[2]>0 and popt[2]<img.shape[1]:
                    cguess = popt[2] # update center pixel

                    xcent_big = np.append(xcent_big, popt[2])
                    ycent_big = np.append(ycent_big, j)
                    wcent_big = np.append(wcent_big, wcent[i])
            except RuntimeError:
                popt = pguess

        # below center line, from middle down
        for j in ydata2:
            yline = img[j-ybuf:j+ybuf, int(pcent[i])-maxbend:int(pcent[i])+maxbend].sum(axis=0)
            # fit gaussian, assume center at 0, width of 2
            if j==ydata2[0]:
                cguess = pcent[i] # xline[np.argmax(yline)]

            pguess = [np.nanmax(yline), img_med, cguess, 2.]
            try:
                popt,pcov = curve_fit(_gaus, xline, yline, p0=pguess)

                if popt[2]>0 and popt[2]<img.shape[1]:
                    cguess = popt[2] # update center pixel

                    xcent_big = np.append(xcent_big, popt[2])
                    ycent_big = np.append(ycent_big, j)
                    wcent_big = np.append(wcent_big, wcent[i])
            except RuntimeError:
                popt = pguess


    if display is True:
        plt.figure()
        plt.imshow(np.log10(img), origin = 'lower',aspect='auto',cmap=cm.Greys_r)
        plt.colorbar()
        plt.scatter(xcent_big,ycent_big,marker='|',c='r')
        plt.show()

    return xcent_big, ycent_big, wcent_big


def find_peaks(wave, flux, pwidth=10, pthreshold=97, minsep=1):
    '''
    given a slice thru a HeNeAr image, find the significant peaks

    :param wave:
    :param flux:
    :param pwidth:
        the number of pixels around the "peak" to fit over
    :param pthreshold:
    Returns
    -------
    Peak Pixels, Peak Wavelengths
    '''
    # sort data, cut top x% of flux data as peak threshold
    flux_thresh = np.percentile(flux, pthreshold)

    # find flux above threshold
    high = np.where((flux >= flux_thresh))

    # find  individual peaks (separated by > 1 pixel)
    pk = high[0][1:][ ( (high[0][1:]-high[0][:-1]) > minsep ) ]

    # offset from start/end of array by at least same # of pixels
    pk = pk[pk > pwidth]
    pk = pk[pk < (len(flux) - pwidth)]

    # print('Found '+str(len(pk))+' peaks in HeNeAr to fit Gaussians to')

    pcent_pix = np.zeros_like(pk,dtype='float')
    wcent_pix = np.zeros_like(pk,dtype='float') # wtemp[pk]
    # for each peak, fit a gaussian to find center
    for i in range(len(pk)):
        xi = wave[pk[i] - pwidth:pk[i] + pwidth]
        yi = flux[pk[i] - pwidth:pk[i] + pwidth]

        pguess = (np.nanmax(yi), np.nanmedian(flux), float(np.nanargmax(yi)), 2.)
        try:
            popt,pcov = curve_fit(_gaus, np.arange(len(xi),dtype='float'), yi,
                                  p0=pguess)

            # the gaussian center of the line in pixel units
            pcent_pix[i] = (pk[i]-pwidth) + popt[2]
            # and the peak in wavelength units
            wcent_pix[i] = xi[np.nanargmax(yi)]

        except RuntimeError:
            pcent_pix[i] = float('nan')
            wcent_pix[i] = float('nan')

    wcent_pix, ss = np.unique(wcent_pix, return_index=True)
    pcent_pix = pcent_pix[ss]
    okcent = np.where((np.isfinite(pcent_pix)))
    return pcent_pix[okcent], wcent_pix[okcent]


def lines_to_surface(img, xcent, ycent, wcent,
                     mode='spline2d', fit_order=2, display=False):
    '''
    Turn traced arc lines into a wavelength solution across the entire chip

    Requires inputs from line_trace(). Outputs are a 2d wavelength solution

    Parameters
    ----------

    img : 2d array
        the HeNeAr data
    xcent : 1d array
        the X (spatial) pixel positions of the HeNeAr lines
    ycent : 1d array
        the Y (wavelength) pixel positions of the HeNeAr lines
    wcent : 1d array
        the wavelength values of the HeNeAr lines
    mode : str, optional
        what mode of interpolation to use to go from traces along the
        HeNeAr lines to a wavelength value for every (x,y) pixel?
        Options include
            poly: along 1-pixel wide slices in the spatial dimension,
                fit a polynomial between the HeNeAr lines. Uses fit_order
            spline: along 1-pixel wide slices in the spatial dimension,
                fit a quadratic spline.
            spline2d: fit a full 2d surface using a cubic spline. This is
                the best option, in principle.

    Returns
    -------
    the 2d wavelenth solution. Output depends on mode parameter.
    '''

    xsz = img.shape[1]

    #  fit the wavelength solution for the entire chip w/ a 2d spline
    if (mode=='spline2d'):
        xfitd = 5 # the spline dimension in the wavelength space
        print('Fitting Spline2d - NOTE: this mode doesnt work well')
        wfit = SmoothBivariateSpline(xcent, ycent, wcent, kx=xfitd, ky=3,
                                     bbox=[0,img.shape[1],0,img.shape[0]], s=0)

    #elif mode=='poly2d':
    ## using 2d polyfit
        # wfit = polyfit2d(xcent_big, ycent_big, wcent_big, order=3)

    elif mode=='spline':
        wfit = np.zeros_like(img)
        xpix = np.arange(xsz)

        for i in np.arange(ycent.min(), ycent.max()):
            x = np.where((ycent == i))

            x_u, ind_u = np.unique(xcent[x], return_index=True)

            # this smoothing parameter is absurd...
            spl = UnivariateSpline(x_u, wcent[x][ind_u], ext=0, k=3, s=5e7)

            if display is True:
                plt.figure()
                plt.scatter(xcent[x][ind_u], wcent[x][ind_u])
                plt.plot(xpix, spl(xpix))
                plt.show()

            wfit[i,:] = spl(xpix)

    elif mode=='poly':
        wfit = np.zeros_like(img)
        xpix = np.arange(xsz)

        for i in np.arange(ycent.min(), ycent.max()):
            x = np.where((ycent == i))
            coeff = np.polyfit(xcent[x], wcent[x], fit_order)
            wfit[i,:] = np.polyval(coeff, xpix)
    return wfit


def load_lines(fname, element):
    '''Takes in .csv file of format wavelength (Angstroms), element, linestrength.
    returns np array of format [[wl, strength], [wl,strength], ...]
    where wl in microns'''
    wls = []
    strengths = []
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                l = line.split(',')
                if l[1].strip(', \n').lower() == element.lower():
                    wls.append(float(l[0].strip(', \n'))/10000)
                    strengths.append(float(l[2].strip(', \n')))
    return np.asarray(wls), np.asarray(strengths)
                
                
            

class lampObs:
    
    def __init__(self, fname):
        
        self.im = image.Image(fname)
        self.header = self.im.header
        self.data = self.im.data
        
    def calibrate(self, flatf, badpxf, biasf = None):
        if biasf != None:
            bias = image.Image(biasf).data
            self.data = self.data - bias
            
        flat = image.Image(flatf).data
        self.data = self.data / flat
        
        badpx = image.Image(badpxf).data
        bad_indices = np.where(badpx == 0)
        smoothed = medfilt(self.data,kernel_size = 7)
        self.data[bad_indices] = smoothed[bad_indices]
        
    def write(self, outfile):
        hdulist_out = self.im.hdulist
        hdulist_out[0].data = self.data
        hdulist_out[0].writeto(outfile, overwrite=True)
        
    def find_lines(self, outdir, element = 'ar', linelist = '/Users/emolter/Python/nirc2_reduce_spec/linelists/nirspec.csv',
               tol=5, fit_order=2, previous='', mode='poly'):
        '''Takes heavily from pydis. Interactively create wavelength solution
               from arc lamp observation'''
        
        warnings.simplefilter('ignore', np.RankWarning)
        
        
        line_wls, line_strengths = load_lines(linelist, element) #microns
        wcen_approx = self.header['CENWAVE'] #um
        wmin_approx, wmax_approx = self.header['MINWAVE'], self.header['MAXWAVE']
        wrange_approx = wmax_approx - wmin_approx
        
        good = np.where(np.logical_and(line_wls < wmax_approx, line_wls > wmin_approx))
        line_wls = line_wls[good]
        line_strengths = line_strengths[good]

        # take a slice thru the data (+/- 10 pixels) in center row of chip
        slice = self.data[self.data.shape[0]/2-10:self.data.shape[0]/2+10,:].sum(axis=0)
        
        ## use the header info to do rough solution (linear guess)
        wtemp = (np.arange(-self.data.shape[0]/2,self.data.shape[0]/2)/float(self.data.shape[0])) * wrange_approx * -1 + wcen_approx
        print(np.min(wtemp), np.max(wtemp))
        
        # = = = = = = = = = = = = = = = =
        #-- manual (interactive) mode
        if (len(previous)==0):
            print('')
            print('Using INTERACTIVE HeNeAr_fit mode:')
            print('1) Click on HeNeAr lines in plot window')
            print('2) Enter corresponding wavelength in terminal and press <return>')
            print('   If mis-click or unsure, just press leave blank and press <return>')
            print('3) To delete an entry, click on label, enter "d" in terminal, press <return>')
            print('4) Close plot window when finished')
        
            xraw = np.arange(len(slice))
            class InteracWave(object):
                # http://stackoverflow.com/questions/21688420/callbacks-for-graphical-mouse-input-how-to-refresh-graphics-how-to-tell-matpl
                def __init__(self):
                    self.fig = plt.figure(figsize = (12,8))
                    self.ax = self.fig.add_subplot(111)
                    self.ax.plot(wtemp, slice/np.max(slice), color = 'b', label = 'data')
                    self.ax.plot(line_wls, line_strengths/np.max(line_strengths), color = 'r', label = 'linelist', marker = '.', linestyle = '')
                    #annotate line list
                    for label, x, y in zip(line_wls, line_wls, line_strengths/np.max(line_strengths)):
                        plt.annotate(str(label), xy=(x, y), xycoords = 'data', ha='center', va='bottom')
                    self.ax.set_xlim([np.min(wtemp), np.max(wtemp)])
                    self.ax.set_xlabel('Wavelength')
                    self.ax.set_ylabel('Counts')
                    plt.legend()
        
                    self.pcent = [] # the pixel centers of the identified lines
                    self.wcent = [] # the labeled wavelengths of the lines
                    self.ixlib = [] # library of click points
        
                    self.cursor = Cursor(self.ax, useblit=False,horizOn=False,
                                         color='red', linewidth=1 )
                    self.connect = self.fig.canvas.mpl_connect
                    self.disconnect = self.fig.canvas.mpl_disconnect
                    self.clickCid = self.connect("button_press_event",self.OnClick)
        
                def OnClick(self, event):
                    # only do stuff if toolbar not being used
                    # NOTE: this subject to change API, so if breaks, this probably why
                    # http://stackoverflow.com/questions/20711148/ignore-matplotlib-cursor-widget-when-toolbar-widget-selected
                    if self.fig.canvas.manager.toolbar._active is None:
                        ix = event.xdata

                        # if the click is in good space, proceed
                        if (ix is not None) and (ix > np.min(wtemp)) and (ix < np.max(wtemp)):
                            # disable button event connection
                            self.disconnect(self.clickCid)
        
                            # disconnect cursor, and remove from plot
                            self.cursor.disconnect_events()
                            self.cursor._update()
        
                            # get points nearby to the click
                            pixscale = wtemp[0] - wtemp[1]
                            nearby = np.where((wtemp > ix-tol*pixscale) &
                                              (wtemp < ix+tol*pixscale))

                            # find if click is too close to an existing click (overlap)
                            kill = None
                            if len(self.pcent)>0:
                                for k in range(len(self.pcent)):
                                    if np.abs(self.ixlib[k]-ix)<tol*pixscale:
                                        kill_d = input('> WARNING: Click too close to existing point. To delete existing point, enter "d"')
                                        if kill_d=='d':
                                            kill = k
                                if kill is not None:
                                    del(self.pcent[kill])
                                    del(self.wcent[kill])
                                    del(self.ixlib[kill])
        
        
                            # If there are enough valid points to possibly fit a peak too...
                            if (len(nearby[0]) > 4) and (kill is None):
                                imax = np.nanargmax(slice[nearby])
        
                                pguess = (np.nanmax(slice[nearby]), np.nanmedian(slice), xraw[nearby][imax], 2.)
                                try:
                                    popt,pcov = curve_fit(_gaus, xraw[nearby], slice[nearby], p0=pguess)
                                    self.ax.plot(wtemp[int(popt[2])], popt[0]/np.max(slice), 'ko', markersize = 4)
                                except ValueError:
                                    print('> WARNING: Bad data near this click, cannot centroid line with Gaussian. I suggest you skip this one')
                                    popt = pguess
                                except RuntimeError:
                                    print('> WARNING: Gaussian centroid on line could not converge. I suggest you skip this one')
                                    popt = pguess
        
                                try:
                                    number=float(input('> Enter Wavelength: '))
                                    self.pcent.append(popt[2])
                                    self.wcent.append(number)
                                    self.ixlib.append((ix))
                                    self.ax.plot(wtemp[int(popt[2])], popt[0]/np.max(slice), 'k+', markersize = 4)
                                    print('Gaussian fit found wl = '+str(wtemp[int(popt[2])])+'. Click again, or if finished close plot window.')
                                except ValueError:
                                    print("> Warning: Not a valid wavelength float!")
        
                            elif (kill is None):
                                print('> Error: No valid data near click!')
        
                            # reconnect to cursor and button event
                            self.clickCid = self.connect("button_press_event",self.OnClick)
                            self.cursor = Cursor(self.ax, useblit=False,horizOn=False,
                                             color='red', linewidth=1 )
                    else:
                        pass
        
            # run the interactive program
            wavefit = InteracWave()
            plt.show() #activate the display - GO!
        
            # how I would LIKE to do this interactively:
            # inside the interac mode, do a split panel, live-updated with
            # the wavelength solution, and where user can edit the fit_order
        
            # how I WILL do it instead
            # a crude while loop here, just to get things moving
        
            # after interactive fitting done, get results fit peaks
            pcent = np.array(wavefit.pcent,dtype='float')
            wcent = np.array(wavefit.wcent, dtype='float')
        
            print('> You have identified '+str(len(pcent))+' lines')
            lout = open(outdir+element+'_lines.dat', 'w')
            lout.write("# This file contains the lines identified [manual] Columns: (pixel, wavelength) \n")
            for l in range(len(pcent)):
                lout.write(str(pcent[l]) + ', ' + str(wcent[l])+'\n')
            lout.close()
        
        
        if (len(previous)>0):
            pcent, wcent = np.loadtxt(previous, dtype='float',
                                      unpack=True, skiprows=1,delimiter=',')
        
        
        #---  FIT SMOOTH FUNCTION ---
        
        # fit polynomial thru the peak wavelengths
        # xpix = (np.arange(len(slice))-len(slice)/2)
        # coeff = np.polyfit(pcent-len(slice)/2, wcent, fit_order)
        xpix = np.arange(len(slice))
        coeff = np.polyfit(pcent, wcent, fit_order)
        wtemp = np.polyval(coeff, xpix)
        
        done = str(fit_order)
        while (done != 'd'):
            fit_order = int(done)
            coeff = np.polyfit(pcent, wcent, fit_order)
            wtemp = np.polyval(coeff, xpix)
        
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(pcent, wcent, 'bo')
            ax1.plot(xpix, wtemp, 'r')
        
            ax2.plot(pcent, wcent - np.polyval(coeff, pcent),'ro')
            ax2.set_xlabel('pixel')
            ax1.set_ylabel('wavelength')
            ax2.set_ylabel('residual')
            ax1.set_title('fit_order = '+str(fit_order))
        
            # ylabel('wavelength')
        
            print(" ")
            print('> If this looks okay, Enter "d" to be done (accept) and then close plot window.')
            print('  If you want to re-fit, enter a number to change the polynomial order and re-fit')
            print('> Currently fit_order = '+str(fit_order))
            print(" ")
        
            plt.show()
        
            _CheckMono(wtemp)
        
            print(' ')
            done = str(input('ENTER: "d" (done) or a # (poly order): '))
            
#write out calibrated data!        
def solve_wl(outfile, lampfiles, linefiles, fmask=(1,), display=False, fit_order = 2, mode = 'poly'):
    '''Given individual wavelength vs pixel line positions for all four lamps
    as output by lampObs.find_lines, generate a single wavelength solution.
    Do line_trace for each, then combine xcent_big, ycent_big, wcent_big 
    before inputting to lines_to_surface.
    inputs are list of lamp image fits files and list of line position files
    output by lampObs.find_lines. List order must correspond'''
    xcent_f = []
    ycent_f = []
    wcent_f = []
    for i in range(len(lampfiles)):
        lampf = lampfiles[i]
        lamp = image.Image(lampf).data
        
        linef = linefiles[i]
        pcent, wcent = np.loadtxt(linef, dtype='float', unpack=True, skiprows=1, delimiter=',')
        
        #-- trace the peaks vertically --
        xcent_big, ycent_big, wcent_big = line_trace(lamp, pcent, wcent,
                                                     fmask=fmask, display=display)

        xcent_f += xcent_big.tolist()
        ycent_f += ycent_big.tolist()
        wcent_f += wcent_big.tolist()
      
    xcent_f = np.asarray(xcent_f)
    ycent_f = np.asarray(ycent_f)
    wcent_f = np.asarray(wcent_f)
        
    #-- turn vertical traces in to a whole chip wavelength solution
    #only uses image itself for shape data
    im = image.Image(lampfiles[0])
    wfit = lines_to_surface(im.data, xcent_f, ycent_f, wcent_f, mode=mode, fit_order=fit_order)
    
    plt.imshow(wfit, origin = 'lower left')
    plt.show()
    
    hdulist_out = im.hdulist
    hdulist_out[0].data = wfit
    hdulist_out[0].writeto(outfile, overwrite=True)
    
