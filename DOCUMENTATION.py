# Instructions for reducing spectroscopic data using this code

#make darks
from nirc2_reduce_spec import darks
fnames = ['raw/internal_cals_spectroscopy/n0122.fits', ...]
dark = darks.Dark(fnames)
dark.write('reduced/internal_cals_spectroscopy/dark_30.fits')


#make flat
from nirc2_reduce_spec import specflat
domeflatoff= ['raw/internal_cals_spectroscopy/n0028.fits', ...]
domeflaton = ['raw/internal_cals_spectroscopy/n0033.fits', ...]
flat = specflat.specFlat(domeflatoff, domeflaton)
tol = 0.07 #how far from median value can a pixel be before it's bad?
blocksize = 4 #must be divisible into 1024. how big of a box to take the median?
flat.make_badpx_map('reduced/internal_cals_spectroscopy/badpx_map_kp.fits',tol,blocksize)
flat.wl_response()
flat.write('reduced/internal_cals_spectroscopy/flat_kp.fits')

#find wavelength solution
from nirc2_reduce_spec import lamp
frame = lamp.lampObs('raw/internal_cals_spectroscopy/n0038.fits')
flatf = 'reduced/internal_cals_spectroscopy/flat_kp.fits'
badpxf = 'reduced/internal_cals_spectroscopy/badpx_map_kp.fits'
frame.calibrate(flatf, badpxf, biasf = None)
outdir = 'reduced/internal_cals_spectroscopy/'
element = 'ar'
frame.write(outdir + 'lamp_'+element+'.fits')
frame.find_lines(outdir, element = element)
#then do the same for helamp, krlamp, nelamp

#put them all together - navigate to reduced directory then run the following
from nirc2_reduce_spec.lamp import solve_wl
elementlist = ['ar','ne','kr','xe']
obsnames = ['lamp_'+el+'.fits' for el in elementlist]
linenames = [el+'_lines.dat' for el in elementlist]
solve_wl('wl_soln.fits', obsnames, linenames)

#apply wl solution to actual science data. from parent dir
from nirc2_reduce_spec import longslit
wl_soln = 'reduced/internal_cals_spectroscopy/wl_soln.fits'
slitim = longslit.Longslit('raw/2017jul02/n0109.fits', wl_soln)
flatf = 'reduced/internal_cals_spectroscopy/flat_kp.fits'
badpxf = 'reduced/internal_cals_spectroscopy/badpx_map_kp.fits'
slitim.calibrate(flatf, badpxf, 'kp', biasf = None)

skyfname = 'reduced/2017jul02/sky_spectrum_phot0.txt'
slitim.plot_med(600, 700) #making sky - choose spot near planet
slitim.save_med(skyfname, 600, 700) #save it
slitim.subtract_sky(skyfname)
#probably need another bad pixel removal step here
slitim.plot_med(300, 400) #should be very near zero
slitim.plot_med(521, 529) #see if planet is reasonable
#slitim.save_med('reduced/2017jul02/northern_storm.txt, 521, 529)
slitim.plot()
slitim.write('reduced/2017jul02/spectrum.fits')


#to open an already calibrated spectrum, just
from nirc2_reduce_spec import longslit
wl_soln = 'reduced/internal_cals_spectroscopy/wl_soln.fits'
slit = longslit.Longslit('reduced/2017jul02/spectrum_phot0.fits', wl_soln)
slit.plot()




# TEST #
from nirc2_reduce_spec import specflat
domeflatoff = ['raw/internal_cals_spectroscopy/n0028.fits', 'raw/internal_cals_spectroscopy/n0029.fits','raw/internal_cals_spectroscopy/n0030.fits','raw/internal_cals_spectroscopy/n0031.fits','raw/internal_cals_spectroscopy/n0032.fits']
domeflaton = ['raw/internal_cals_spectroscopy/n0033.fits', 'raw/internal_cals_spectroscopy/n0034.fits','raw/internal_cals_spectroscopy/n0035.fits','raw/internal_cals_spectroscopy/n0036.fits','raw/internal_cals_spectroscopy/n0037.fits']
flat = specflat.specFlat(domeflatoff, domeflaton)
flat.wl_response()
flat.write('reduced/internal_cals_spectroscopy/flat.fits)