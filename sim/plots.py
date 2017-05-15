#!/usr/bin/python
# Create plots from sims

import numpy as np
import pickle
import matplotlib.pyplot as plt

results_dir='results_UDS/'
nruns = 1
rsearch = 9 # how far away to consider source from SC2 peak, arcsec

axis_range = [0,20,-0.1,2.1]
#symbol = 'ko'
symbsize=5
linesymbol = 'k--'
linesymbolcut = 'k:'

sc2cut = 8

print 'Combine data from sims...'
stats = []
allmapnoise = []
for i in range(nruns):
    with open(results_dir+'sim_'+str(i)+'.pickle', 'rb') as f:
        data = pickle.load(f)
    stats.extend(data['stats'])
    mapnoise = data['mapnoise']
    print mapnoise
    allmapnoise.append(mapnoise)

with open(results_dir+'psf.pickle', 'rb') as f:
    d = pickle.load(f)
pixres = d['pixres']
psf = d['image']

mapnoise = np.mean(allmapnoise)
print "Mean SCUBA-2 map noise is: ",mapnoise,'mJy'

smax_in = []
rmax_in = []
xmax_in = []
ymax_in = []
for rec in stats:
    # find brightness and distance of brightest input catalogue source
    # within rsearch
    x = rec['inputs']['x']
    y = rec['inputs']['y']
    r = np.sqrt(x**2 + y**2)
    rec['inputs']['r'] = r
    inradius = np.where(r <= rsearch)
    s_search = rec['inputs']['f'][inradius]
    x_search = rec['inputs']['x'][inradius]
    y_search = rec['inputs']['y'][inradius]
    r_search = r[inradius]
    argmax = s_search.argmax()
    smax_in.append(s_search[argmax])
    xmax_in.append(x_search[argmax])
    ymax_in.append(y_search[argmax])
    rmax_in.append(r_search[argmax])

smax_in = np.array(smax_in)
xmax_in = np.array(xmax_in)
ymax_in = np.array(ymax_in)
rmax_in = np.array(rmax_in)


#smax_in = np.array([max(r['inputs']['f']) for r in stats])

s_sc2 = np.array([r['peak']['f'] for r in stats])
s_sc2_noiseless = np.array([r['peak']['f_noiseless'] for r in stats])


# Plot ratio of maximum nearby input flux / SCUBA-2 flux, as a function
# of SCUBA-2 flux.

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter( s_sc2, smax_in/s_sc2 )
ax.plot( axis_range[0:2], [1.,1.], linesymbol )
ax.plot( [sc2cut,sc2cut], axis_range[2:4], linesymbolcut )
ax.set_xlabel('Measured peak SCUBA-2 peak flux (mJy)')
ax.set_ylabel('Maximum input flux / peak SCUBA-2 flux')
plt.axis(axis_range)
plt.savefig('fluxes_input_sc2.png', dpi=300)


# Plot ratio of noiseless SCUBA-2 map flux to SCUBA-2 flux, as a function of
# SCUBA-2 flux

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter( s_sc2, s_sc2_noiseless/s_sc2 )
ax.plot( axis_range[0:2], [1.,1.], linesymbol )
ax.plot( [sc2cut,sc2cut], axis_range[2:4], linesymbolcut )
ax.set_xlabel('Measured peak SCUBA-2 peak flux (mJy)')
ax.set_ylabel('Noiseless peak SCUBA-2 flux / peak SCUBA-2 flux')
plt.axis(axis_range)
plt.savefig('fluxes_noiseless_sc2.png', dpi=300)

#for i in range(len(stats)):
#    r = stats[i]
#    print '--- '+str(i)+' ---'
#    print 'SCUBA2:'+str(r['peak'])
#    print '   SMA:'+str(r['sma'])
#    print 'inputs:'+str(r['inputs'])


# compare SMA flux and position to brightest input source
smax_sma = []
rmax_sma = []
xmax_sma = []
ymax_sma = []
for rec in stats:

    x = rec['sma']['x']
    y = rec['sma']['y']

    stemp = 0
    xtemp = -1
    ytemp = -1
    rtemp = -1

    if len(x) != 0:
        r = np.sqrt(x**2 + y**2)
        rec['sma']['r'] = r
        inradius = np.where(r <= rsearch)

        if len(inradius[0]) != 0:
            s_search = rec['sma']['f'][inradius]
            x_search = rec['sma']['x'][inradius]
            y_search = rec['sma']['y'][inradius]
            r_search = r[inradius]
            argmax = s_search.argmax()
            stemp = s_search[argmax]
            xtemp = x_search[argmax]
            ytemp = y_search[argmax]
            rtemp = r_search[argmax]

    smax_sma.append(stemp)
    rmax_sma.append(rtemp)
    xmax_sma.append(xtemp)
    ymax_sma.append(ytemp)

smax_sma = np.array(smax_sma)
xmax_sma = np.array(xmax_sma)
ymax_sma = np.array(ymax_sma)
rmax_sma = np.array(rmax_sma)

#print rmax_sma

#ind = np.where(xmax_sma != -1)
ind = np.where(smax_sma >= 0)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter( s_sc2[ind], smax_sma[ind]/smax_in[ind] )
ax.plot( axis_range[0:2], [1.,1.], linesymbol )
ax.plot( [sc2cut,sc2cut], axis_range[2:4], linesymbolcut )
ax.set_xlabel('Measured peak SCUBA-2 peak flux (mJy)')
ax.set_ylabel('Maximum SMA flux / peak input flux')
plt.axis(axis_range)
plt.savefig('fluxes_sma_input.png', dpi=300)


# what fraction of the noiseless sc2 map flux is represented by the
# brightest input flux as a function of sc2 flux

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter( s_sc2, smax_in/s_sc2_noiseless )
ax.plot( axis_range[0:2], [1.,1.], linesymbol )
ax.plot( [sc2cut,sc2cut], axis_range[2:4], linesymbolcut )
ax.set_xlabel('Measured peak SCUBA-2 peak flux (mJy)')
ax.set_ylabel('Maximum input flux / Noiseless peak SCUBA-2 flux')
plt.axis(axis_range)
plt.savefig('fluxes_in_noiseless.png', dpi=300)


# same thing, but using SMA fluxes to find brightest contributor
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#ax.plot( s_sc2[ind], smax_sma[ind]/s_sc2_noiseless[ind],symbol )
plt.scatter( s_sc2[ind], smax_sma[ind]/s_sc2_noiseless[ind])
ax.plot( axis_range[0:2], [1.,1.], linesymbol )
ax.plot( [sc2cut,sc2cut], axis_range[2:4], linesymbolcut )
ax.set_xlabel('Measured peak SCUBA-2 peak flux (mJy)')
ax.set_ylabel('Maximum SMA flux / Noiseless peak SCUBA-2 flux')
plt.axis(axis_range)
plt.savefig('fluxes_sma_noiseless.png', dpi=300)


# Plot distance of brightest SMA peak from brightest input source
r_bright = []
for i in range(len(xmax_sma)):
    if xmax_sma[i] != -1:
        r = np.sqrt((xmax_sma[i]-xmax_in[i])**2 + \
                    (ymax_sma[i]-ymax_in[i])**2)
    else:
        r = -1
    r_bright.append(r)

r_bright = np.array(r_bright)


thisrange = [axis_range[0],axis_range[1],-2,15]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter( s_sc2[ind], r_bright)
ax.plot( [sc2cut,sc2cut], thisrange[2:4], linesymbolcut )
ax.set_xlabel('Measured peak SCUBA-2 peak flux (mJy)')
ax.set_ylabel('Offset between SMA peak and input source peak (arcsec)')
plt.axis(thisrange)
plt.savefig('fluxes_sma_offsets.png', dpi=300)

# plot offset between SMA and SCUBA-2 peaks
r_bright = []
for i in range(len(xmax_sma)):
    if xmax_sma[i] != -1:
        r = np.sqrt((xmax_sma[i]-xmax_in[i])**2 + \
                    (ymax_sma[i]-ymax_in[i])**2)
    else:
        r = -1
    r_bright.append(r)

r_bright = np.array(r_bright)


thisrange = [axis_range[0],axis_range[1],-2,15]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter( s_sc2[ind]/mapnoise, rmax_sma)
ax.plot( [sc2cut,sc2cut], thisrange[2:4], linesymbolcut )
ax.set_xlabel('SCUBA-2 SNR')
ax.set_ylabel('Offset of brightest SMA peak from SCUBA-2 (arcsec)')
plt.axis(thisrange)
plt.savefig('sma_sc2_offsets.png', dpi=300)

# plot input flux density vs. SMA flux density
thisrange = [0,25,0,25]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter( smax_sma[ind], smax_in[ind] )
ax.plot( thisrange[0:2], thisrange[2:4], linesymbol )
#ax.plot( [sc2cut,sc2cut], axis_range[2:4], linesymbolcut )
ax.set_xlabel('SMA flux (mJy)')
ax.set_ylabel('True flux (mJy)')
plt.axis('equal')
plt.axis(thisrange)
plt.savefig('sma_flux_boosting.png', dpi=300)
