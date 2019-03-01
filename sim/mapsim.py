#!/usr/bin/env python3
# Create a map of point sources with fluxes drawn from counts
# distribution

import numpy as np
import pickle
import matplotlib.pyplot as plt
from astropy.convolution import convolve, convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.io import fits
from scipy import interpolate
import sys
from drawdist1d import drawdist1d
from photutils.detection import find_peaks
from photutils import CircularAperture
import os

# display
plotsources = False

# how many runs to do / where to store results
nruns = 1
results_dir = 'results_UDS/'

# define some constants for the simulation
fmin = 0.1    # min flux to consider in extrapolation
nci  = 10     # number of points in interpolated integral counts
pixres = 2    # arcsec / pixel scuba-2 simulated map
#n      = 1764  # number of pixels on a side of square scuba-2 simulated map
n=900
#sc2_fudge = 6.13 # fudge factor for raw pixel noise in SC2 maps
#CLS map noise is average of field noises weighted by areas
sc2_noise = 0.9 # mJy noise in smoothed source-finding SC2 maps
#Faintest source folowed up in our survey
flux_thresh = 8.0 #10 # scuba-2 flux threshold to find sources
plot_thresh = 1  # scuba-2 flux threshold to plot input sources

area = (n*pixres/3600.)**2. # area of scuba-2 simulated map

sma_pixres = 0.1               # arcsec / pixel for sma maps
sma_thumb = 51.2/sma_pixres      # pixels on side of SMA thumbnail
sma_fwhm = 2.4/sma_pixres       # pixels
#Mean noise from each pointing
sma_noise = 1.3              # SMA noise in mJy
#sma_fudge = 7.59               # fudge factor raw pixel noise SMA maps
rsearch = 9                    # arcsec search radius

sma_thumb = int(sma_thumb)
sma_thresh = sma_noise*4.      # SMA detection threshold

# image intensity clipping
vmin=-5
vmax=10

# load in images --------------------------------------------------------------

try:
    os.makedirs(results_dir)
except Exception as e:
    pass

psf_file = '../cls/psf850_norm.fits'
rms_file = '../cls/wcs_f_UDS-CANDELS_8_rms.fits'

h = fits.open(psf_file)
psf = h[0].data
h.close()

#h = fits.open(rms_file)
#rms = h[0].data
#h.close()


sma_psf = Gaussian2DKernel(stddev=sma_fwhm/(2*np.sqrt(2*np.log(2))))

# write out pickled psf
with open(results_dir+'/psf.pickle','wb') as outfile:
    pickle.dump({'image':psf,'pixres':pixres}, outfile,
                pickle.HIGHEST_PROTOCOL)

# counts from Simpson et al. 2015 -------------------------------------------

#def calcscale(x):
#    m = (x[1][0] - x[0][0]) / float(x[1][1] - x[0][1])
#    b = x[0][0] - m*x[0][1]
#
#    return m, b
#
#fscale = [ (0,113), (1,542) ] # log flux
#cscale = [ (0,414), (5,33) ] # log cumulative counts
#samples = [(114,153), (209,186), (287,211), (381,244), (450,267), (480,284),
#           (501,303), (523,323), (550,348), (575,372), (597,394), (618,410),
#           (712,452)]
#
#fm,fb = calcscale(fscale)
#cm,cb = calcscale(cscale)
#
#f = 10.**np.array([i[0]*fm + fb for i in samples] )
#ci = 10.**np.array([i[1]*cm + cb for i in samples] )
#
## extrapolate to small flux values
#m = (np.log10(ci[1])-np.log10(ci[0])) / (np.log10(f[1])-np.log10(f[0]))
#b = np.log10(ci[0]) - m*np.log10(f[0])
#
#f = np.append( [fmin], f )
#ci = np.append( [10.**(m*np.log10(fmin) + b)], ci )
#
## interpolate integral counts on to a finer grid
#f_interp = interpolate.interp1d(np.log10(f),np.log10(ci))
#
#f = 10.**np.linspace(np.log10(min(f)),np.log10(max(f)),num=nci)
#ci = 10.**f_interp(np.log10(f))
#
#
## calculate the differential counts in logarithmic bins to use as pdf
#logfbin = np.log10(f)
#cd = ci[0:-1] - ci[1:]

Smin=0.1
Smax=40

def prior_count_casey(S):
    gamma=1.4
    S0=3.7
    N0=3300.
    if isinstance(S,int) or isinstance(S,float):
        if (S > Smin) and (S < Smax):
            p=(N0/S0)*(S/S0)**-gamma*np.exp(-S/S0)
        else:
            p=0
    else:
        p=[]
        for s in S:
            if (s > Smin) and (s < Smax):
                p.append((N0/S0)*(s/S0)**-gamma*np.exp(-s/S0))
            else:
                p.append(0)
        p=np.array(p)
    return p
    
dfbin=0.1
fbin=np.array([Smin+i*dfbin for i in range(int(round((Smax-Smin)/dfbin+1)))])
cd=prior_count_casey([0.5*(fbin[i]+fbin[i+1]) for i in range(len(fbin)-1)])

# -----------------------------------------------------------------------------

for run in range(nruns):

    print(run,'/',nruns,'----------------------------------------------')

    thumbsdir = results_dir+'/thumbs'+str(run)
    try:
        os.makedirs(thumbsdir)
    except Exception as e:
        pass

    #print(max(ci), area)
    ngal = int(np.sum(cd)*dfbin*area)
    print('ngal > ',fmin,':',ngal)

    # draw fluxes from distribution
    f = drawdist1d(cd,fbin,ngal)

    # choose random positions
    x_pix = np.random.uniform(0, n-1, ngal)
    y_pix = np.random.uniform(0, n-1, ngal)
    x = x_pix*pixres # arcsec
    y = y_pix*pixres


    # get x,y,f of max value as a check
    imax = np.argmax(f)
    print('max pixel coordinates:',x[imax]/pixres,y[imax]/pixres,f[imax])

    rawmap = np.zeros((n,n))
    psf_smooth = convolve(psf,psf,normalize_kernel=False)

    for i in range(ngal):
        rawmap[int(y[i]/pixres),int(x[i]/pixres)] += f[i]

    map_beam = convolve_fft(rawmap, psf, normalize_kernel=False)
    map_smooth_noiseless = convolve_fft(map_beam, psf, normalize_kernel=False)

    map_noise = np.random.normal(size=(n,n))*sc2_noise*np.sqrt(np.max(psf_smooth))
    map_beam += map_noise #np.random.normal(size=(n,n))*noise

    map_smooth = convolve_fft(map_beam, psf, normalize_kernel=False)
    map_smooth = map_smooth / np.max(psf_smooth)
    map_smooth_noiseless = map_smooth_noiseless / np.max(psf_smooth)
    noise_smooth = convolve_fft(map_noise, psf, normalize_kernel=False) / np.max(psf_smooth)

    #scale = np.sqrt(1./(1.1**2.*np.sum(psf**2)))
    #psf = scale*psf
    #test1 = np.random.normal(size=(n,n))
    #test2 = convolve_fft(test1,psf,normalize_kernel=False)
    #test_noise = convolve_fft(test2,psf,normalize_kernel=False)
    #print 'smooth noise map RMS, var:',np.std(test_noise), np.var(test_noise)
    #print 'sum psf, psf**2:', np.sum(psf), np.sum(psf**2)
    #print 'psf and map dimensions:', len(psf), len(test_noise)
    #m1 = fits.PrimaryHDU(test1)
    #m1.writeto('m1.fits', overwrite=True)
    #m2 = fits.PrimaryHDU(test2)
    #m2.writeto('m2.fits', overwrite=True)
    #m3 = fits.PrimaryHDU(test_noise)
    #m3.writeto('m3.fits', overwrite=True)
    #sys.exit(1)
    
    print("stuff:",np.max(rawmap),np.max(map_beam),np.max(map_smooth))
    map_rms = np.std(noise_smooth)
    print("smooth map noise rms:", map_rms)

    # find peaks
    peaks = find_peaks(map_smooth,flux_thresh,border_width=15)

    # generate SMA maps around these peaks. Descending flux order
    count = 0
    stats = []

    pi_sorted = np.argsort([p['peak_value'] for p in peaks])[::-1]
    #for p in peaks:
    for pi in pi_sorted:
        p = peaks[pi]
        p_xpix,p_ypix,p_f = p['x_peak'],p['y_peak'],p['peak_value']

        # noiseless peak may be slightly offset. Look in box ns x ns
        ns = 2
        submap = map_smooth_noiseless[p_ypix-ns:p_ypix+ns+1,
                                      p_xpix-ns:p_xpix+ns+1]
        p_f_noiseless = np.max(submap)
        #print 'test:',map_smooth_noiseless[p_ypix,p_xpix], p_f_noiseless

        p_x = p_xpix*pixres  # convert to arcsec from pixels
        p_y = p_ypix*pixres

        print(count,'peak pixel',p_xpix, p_ypix, p_f, p_f_noiseless)

        sma_raw = np.zeros((sma_thumb, sma_thumb))
        all_i = [] # indices of input sources from master list in thumb
        all_x_thumb = []
        all_y_thumb = []
        all_f_thumb = []
        for i in range(ngal):
            x_thumb = int(x[i] - p_x)/sma_pixres + sma_thumb/2
            y_thumb = int(y[i] - p_y)/sma_pixres + sma_thumb/2
            f_thumb = f[i]

            # Add flux to simulated SMA map if within thumbnail
            if x_thumb >= 0 and x_thumb < sma_thumb and \
               y_thumb >= 0 and y_thumb < sma_thumb:
                #print "  ",x[i]/pixres, y[i]/pixres, f_thumb
                sma_raw[int(y_thumb),int(x_thumb)] += f_thumb
                all_x_thumb.append(x_thumb)
                all_y_thumb.append(y_thumb)
                all_f_thumb.append(f_thumb)
                all_i.append(i)

        # Print input sources in thumb sorted by descending flux
        i_sorted = np.argsort(f[all_i])[::-1]
        for i in i_sorted:
            j = all_i[i]
            #print "  ",x[j]/pixres, y[j]/pixres, f[j]

        # Create an SMA noise map with the right RMS
        sma_noisemap = convolve(np.random.normal(size=(sma_thumb,sma_thumb)),
                                    sma_psf)*sma_noise/np.sqrt(np.max(convolve(sma_psf,sma_psf)))

        # Create a smooth SMA signal map with the noise added to it
        sma_smooth = convolve(sma_raw, sma_psf)/np.max(sma_psf) + sma_noisemap
        h_sma = fits.PrimaryHDU(sma_smooth)
        #Changed 'overwrite' to 'clobber', using older Astropy version
        h_sma.writeto(thumbsdir+'/sma'+str(count)+'.fits', clobber=True)

        # Plot of the simulated SMA map with input sources shown
        fig = plt.figure(figsize=(4.5,4.5), dpi=100)
        plt.imshow(sma_smooth, cmap='gray', interpolation='none',
                   vmin=vmin,vmax=vmax, origin='lower',
                   extent=np.array([-0.5,0.5,-0.5,0.5])*sma_thumb*sma_pixres)

        all_x_thumb = np.array(all_x_thumb)
        all_y_thumb = np.array(all_y_thumb)
        all_f_thumb = np.array(all_f_thumb)

        # identify sources brighter and fainter than thresh. Those input
        # sources fainter than the threshold shown in yellow, those brighter
        # in red.
        #thresh = p_f / 3.
        thresh = sma_noise
        #print thresh
        #thresh = sma_thresh
        i_bright = all_f_thumb >= thresh
        i_faint = all_f_thumb < thresh

        #Red points brighter than SMA noise
        plt.scatter((all_x_thumb[i_bright]-0.5*sma_thumb)*sma_pixres,
                    (all_y_thumb[i_bright]-0.5*sma_thumb)*sma_pixres,
                    s=15,
                    facecolors='none', edgecolors='r')
        #Yellow points fainter than SMA noise
        plt.scatter((all_x_thumb[i_faint]-0.5*sma_thumb)*sma_pixres,
                    (all_y_thumb[i_faint]-0.5*sma_thumb)*sma_pixres,
                    s=5,
                    facecolors='none', edgecolors='y')


        # find SMA peaks and show in blue
        peaks_thumb = find_peaks(sma_smooth,sma_thresh,border_width=3)
        sma_x = [p['x_peak'] for p in peaks_thumb]
        sma_y = [p['y_peak'] for p in peaks_thumb]
        sma_f = [p['peak_value'] for p in peaks_thumb]
        for p in peaks_thumb:
            #Blue circles are detected peaks found in SMA image
            plt.scatter((p['x_peak']-0.5*sma_thumb)*sma_pixres,
                        (p['y_peak']-0.5*sma_thumb)*sma_pixres,
                        s=100,
                        facecolors='none',
                        edgecolors='b')
            plt.text((p['x_peak']+5-0.5*sma_thumb)*sma_pixres,
                     (p['y_peak']+5-0.5*sma_thumb)*sma_pixres,
                     '%4.2f' % p['peak_value'],
                     color='w')

        # annotation and save
        plt.title('%i: (%i,%i), $S_{SC2}$=%4.2f mJy' % \
                  (count,p_x-0.5*n*pixres,p_y-0.5*n*pixres,p_f))
        ax = plt.gca()
        ax.set_xlabel('$\delta X$ (arcsec)')
        ax.set_ylabel('$\delta Y$ (arcsec)')
        ax.set_xlim(np.array([-0.5,0.5])*sma_thumb*sma_pixres)
        ax.set_ylim(np.array([-0.5,0.5])*sma_thumb*sma_pixres)
        plt.savefig(thumbsdir+'/sma'+str(count)+'.png')

        if plotsources:
            plt.show()

        plt.close()

        # create a record for calculating stats at the end
        #   peak   = SCUBA-2 peak    (SCUBA-2 simulated map arcsec coords)
        #   inputs = input catalogue (input catalogue off from SCUBA-2 arcsec)
        #   sma    = sma outputs     (SMA simulated map offsets arcsec)
        record = {'peak'  : {'x':p_x, 'y':p_y, 'f':p_f,
                             'f_noiseless':p_f_noiseless},
                  'inputs': {'x':x[all_i] - p_x,
                             'y':y[all_i] - p_y,
                             'f':f[all_i]},
                  'sma'   : {'x':(np.array(sma_x) - sma_thumb/2)*sma_pixres,
                             'y':(np.array(sma_y) - sma_thumb/2)*sma_pixres,
                             'f':np.array(sma_f)}}
        stats.append(record)

        count += 1

    # write out pickled results
    with open(results_dir+'/sim_'+str(run)+'.pickle','wb') as outfile:
        pickle.dump({'stats':stats,'mapnoise':map_rms}, outfile,
                    pickle.HIGHEST_PROTOCOL)


    for whichmap in ['','_noiseless']:
        if whichmap is '':
            themap = map_smooth
            title = 'Simulated SCUBA-2 map, noise RMS=%4.2f mJy' % (map_rms)
        else:
            themap = map_smooth_noiseless
            title = 'Simulated noiseless SCUBA-2 map'

        # create new FITS file
        h_new = fits.PrimaryHDU(themap)
        #Changed 'overwrite' to 'clobber', using older Astropy version
        h_new.writeto(results_dir+'/sim_'+str(run)+whichmap+'.fits',
                      clobber=True)

        # Plot of the simulated map with found peaks and input sources
        fig = plt.figure(figsize=(8,8), dpi=100)
        plt.imshow(themap, cmap='gray', interpolation='none',
                        origin='lower',
                        vmin=vmin, vmax=vmax,
                        extent=np.array([-0.5,0.5,-0.5,0.5])*n*pixres)
        #plt.imshow(map_beam, cmap='gray', interpolation='none')
        #Plot sources with S>cutoff in red
        plt.scatter((peaks['x_peak']-0.5*n)*pixres,
                    (peaks['y_peak']-0.5*n)*pixres,
                    s=100*pixres,
                    facecolors='none',
                    edgecolors='r')
        #Plot sources with S>1mJy in blue
        i = f > 1
        plt.scatter((x_pix[i]-0.5*n)*pixres,
                    (y_pix[i]-0.5*n)*pixres,
                    s=4*pixres,
                    facecolors='none',
                    edgecolors='b')

        plt.title(title)
        ax = plt.gca()
        ax.set_xlabel('$X$ (arcsec)')
        ax.set_ylabel('$Y$ (arcsec)')
        ax.set_xlim(np.array([-0.5,0.5])*n*pixres)
        ax.set_ylim(np.array([-0.5,0.5])*n*pixres)
        plt.savefig(results_dir+'/sim_'+str(run)+whichmap+'.png')

        if plotsources:
            plt.show()





#x = np.random.random((256,256))


#plt.imshow(x, cmap='gray', interpolation='none')
#plt.show()
