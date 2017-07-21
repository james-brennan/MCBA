"""
algorithms.py

Implementations of BA algorithms

"""
import os
import pylab as plt
import numpy as np
import h5py
import numpy as np
import copy
from scipy import ndimage
import skimage.filters
from scipy import ndimage
import skimage.measure
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
import pylab as plt
import datetime


def MCD64(r, hs, W=10, stdSmaxr=8, dBThreshold=5, sigmap=5):

   """

   parameters:

       W: window length (default 10) (int)

       stdSmaxr: maximum local std for pixels to be burnt.
           default is 8 days above which not burnt (float)

       dBThreshold: distance away from hotspot
           for initial classification of pixel as
               unburnt (int)

       sigmap: distance decay away from burnt pixels
               typically 5km (float)


   """
   b5 = r[:, 4]
   b7 = r[:, 6]
   VI = (b5 - b7 )/( b5 + b7)

   """
   Temporal change index
   ---------------------
   Two rolling windows seperated
   by ten days.
   difference between two divided
   by standard deviation of both
   """
   #W = 10                  # window size
   N = 366                 # theoretical number of obs
   N_WINDOWS = N-2*W+ 1    # number of 10 day rolling composites
   # figure out centre day of these windows
   doy = np.arange(W, N-W+1)

   xsize = r.shape[2]
   ysize = r.shape[3]

   # make arrays to store
   VI_pre = np.ones((N_WINDOWS, xsize,  ysize))
   sigma_pre = np.ones((N_WINDOWS, xsize,  ysize))
   # and post
   VI_post = np.ones((N_WINDOWS,  xsize,  ysize))
   sigma_post = np.ones((N_WINDOWS,  xsize,  ysize))
   """
   Define temporal seperabilty metric $S$
       S(x,y,k) = (dVI(x,y,k)) / ( 0.5 * [sigma_pre(x,y,k) + sigma_post(x,y,k)] )
   where dVI(x,y,k) = VI_pre(x,y,k) - VI_post(x,y,k)
   """
   # run it over the grid
   """
   Run for this pixel
   -- can probably vectorise? with axis to trim mean etc
   paper says that , the smallest tenth and largest tenth of
   the observations are excluded so can actually just take off
   the smallest and largest value...
   """

   import copy
   def doTrimmed(data):
       dat = np.copy(data)
       dat.sort(axis=0)
       # remove top and bottom is 10%
       std = np.ma.mean(dat[1:int(W)], axis=0)
       mean = np.ma.std(dat[1:int(W)], axis=0)
       # now for locations that
       # have less than three obs
       #count = np.sum(data>0, axis=0)
       # remove these...
       #has_enough = count > 2
       return std, mean



   # do pre
   k0 = 0
   for nk in xrange(N_WINDOWS):
       pre = VI[k0 : k0+W-1, :, :]
       premean, prestd = doTrimmed(pre)
       VI_pre[nk] = premean
       sigma_pre[nk] = prestd
       # and post fire window
       post = VI[k0+W-1 : k0+2*W-1, :, :]
       postmean, poststd = doTrimmed(post)
       VI_post[nk] = postmean
       sigma_post[nk] = poststd
       # increment k0 by one day
       k0 += 1

   # prepare result
   S = ((VI_pre - VI_post) / (sigma_pre + sigma_post )/2)

   """
   S is the temporal metric
   Now want to find date of maximum S
   """
   Smax = np.argmax(S, axis=0)
   # Now temporal texture...
   # 3x3 neighbourhood standard deviation
   from scipy import ndimage
   import skimage.filters
   result = ndimage.generic_filter(Smax, np.std, size = 3)
   selem = np.ones((3,3)).astype(int)
   stdSmaxr = skimage.filters.rank.percentile(result.astype(int), selem=selem)


   """
   Create definite unburnt
   """
   unburnt = np.logical_and(Smax < 2, stdSmaxr > 8)
   """
   active fire date mask
   """
   hsdates = np.argwhere(hs).T
   tf = np.zeros((xsize, ysize))
   tf[hsdates[1], hsdates[2]] = hsdates[0]
   # figure out burnt for sure pixels
   condition1 = np.logical_and((Smax - tf) < 15, tf >0)
   condition2 = Smax >= 2
   condition3 = stdSmaxr <= 8

   burnt = condition1 & condition2 & condition3

   """
   Some region growing thing now
   1. cluster each set
   2. only grow clusters larger than 50
   """
   import skimage.measure
   scars = skimage.measure.label(burnt, neighbors=8)
   """
   now look at each scar and
   1. check it has a CBP
   2. meets threshold test
   """

   # some setup
   deltaVI = VI_pre - VI_post
   # get maximum sep one
   dVIstar = np.zeros((xsize, ysize))
   VI_poststar = np.zeros((xsize, ysize))

   for x in xrange(xsize):
       for y in xrange(ysize):
           dVIstar[x, y] = deltaVI[Smax[x,y], x,y]
           VI_poststar[x, y] = VI_post[Smax[x,y], x,y]


   burnt = np.zeros((xsize, ysize))

   for sc in xrange(1, scars.max()):
       scc = scars==sc
       # check it has a CBP
       """
       is it larger than 50 pixels?
       """
       if scc.sum() >= 50:
           #scc = scc.astype(int)
           """
           Region grow this cluster
           -- a bit like HANDS here
           """
           prevSum = scc.sum()
           growing = True
           g = []
           k = 0
           prev_edge = np.zeros((xsize, ysize)).astype(bool)
           while growing:
               gg = np.gradient(scc)
               edge = gg[0]**2 + gg[1]**2 > 0
               edge = np.logical_and(edge, ~prev_edge)
               edge = np.logical_and(edge, ~scc)
               prev_edge = np.copy(edge)
               g.append(edge)
               """
               check conditions for these pxls
               """
               cond8 = dVIstar[edge] >  np.percentile(dVIstar[scc], 25)
               cond9 = VI_poststar[edge] < np.percentile(VI_poststar[scc], 75)
               cond10 = stdSmaxr[edge] <= 3
               #cond11 =
               conditions = cond8 & cond9 & cond10
               if conditions.sum() > 0:
                   # we can region grow
                   edge[(edge==True)]=conditions
                   scc[edge] = True
                   prevSum = scc.sum()
               else:
                   growing = False
               k += 1

       """
       save these
       """
       burnt[scc]=True

   burnt = burnt.astype(bool)
   """
   calculate distances
   -- need this for unburnt conditional density
   and also Pb
   """
   fx, fy = np.where(burnt)
   fire_locs = np.vstack([fx, fy]).T
   vor = cKDTree( fire_locs )
   xy = np.mgrid[0:xsize:1, 0:ysize:1].reshape(2, -1)

   nn = vor.query(xy.T, k=1)
   dB = nn[0].reshape((xsize, ysize))*0.5

   unburnt = np.logical_or(unburnt, dB>dBThreshold)
   """
   classification part
   make kde likelihood functions
   """
   import scipy.stats
   lBurnt = scipy.stats.gaussian_kde(dVIstar[burnt])
   lUnBurnt = scipy.stats.gaussian_kde(dVIstar[unburnt])

   """
   now these take AGES to evaluate the pdf because
   we have to many samples (espc. unburnt).
   Instead we sample from these to generate a smaller
   sample for the pdf() evaluation
   """
   unburnt_sample = lUnBurnt.resample(1000).reshape(-1)
   burnt_sample = lBurnt.resample(1000).reshape(-1)
   # re-generate likelihood functions
   lUnBurnt = scipy.stats.gaussian_kde(unburnt_sample)
   lBurnt = scipy.stats.gaussian_kde(burnt_sample)
   """
   Do priors
   pB
   main thing is distance to nearest burnt training
   pixel -- done this above
   """

   """
   equation 15
   pB(x,y) = (Pmax - Pmin) * exp(-dist/2sigma**2) + Pmin
   """
   #sigmap = 5
   Pmax = 0.5
   Pmin = 0.002
   pb = (Pmax-Pmin)*np.exp(-dB/(2*sigmap**2)) + Pmin
   pu = 1-pb
   """
   compute P(B|VI)
   """
   likelihoodBurnt = lBurnt.pdf(dVIstar.flatten()).reshape((xsize, ysize))
   likelihoodUnBurnt = lUnBurnt.pdf(dVIstar.flatten()).reshape((xsize, ysize))
   PB = likelihoodBurnt * pb / (likelihoodBurnt * pb  + likelihoodUnBurnt *pu)
   """
   initial classification
   """
   cond1 = ~unburnt
   cond2 = PB > 0.6
   VIpost98 = np.percentile(VI_poststar[burnt], 98)
   sigmat98 = np.percentile(stdSmaxr[burnt], 98)
   cond3 = VI_poststar <= VIpost98
   cond4 = stdSmaxr <= sigmat98
   cc   = cond1 & cond2 & cond3 & cond4
   newUnburnt = ~cc
   newBurnt = cc
   """
   temporal consistency filter
   """
   def timeCheck(arr):
       mid = arr[4]
       boold = np.abs(mid - arr)
       boold = boold <= 5
       if boold.sum() < 3:
           return 0
       elif boold.sum() > 6:
           return 1
       else:
           return 2
   #tt = ndimage.generic_filter(Smax, timeCheck, size = 3)
   """
   unburned pixels need 1 to be reclassifed as burnt
   burnt pixels need 0 to be reclassifed as unburt
   else no change
   """
   #addToBurnt = np.logical_and(tt==1, newUnburnt)
   #addToUnBurnt = np.logical_and(tt==0, newBurnt)

   dob = np.zeros((xsize, ysize))
   dob[newBurnt] =  doy[Smax[newBurnt]]
   return dob



def HANDS(r, hs, W=10, gstd=1):
   """
   Algorithm similar to HANDS
   algorithm
   Use active fires to train
   threshold in delta NDVI

   two parameters:

       W: window length -- default is 10 days (int)
       gstd: number of standard deviations from mean for burnt pixel (float)

   """
   from scipy import ndimage

   # window length
   #W = 10
   # number of std for ndvi
   #gstd = 1
   xsize = r.shape[2]
   ysize = r.shape[3]
   def modeFilter(arr):
       if arr.sum() > 5:
           return 1
       else:
           return 0

   ndvi = (r[:, 1] - r[:, 0])/(r[:, 1] + r[:, 0])
   # make 10 day maximum value composite
   MVC_idxs = []
   maxNDVI = []
   for day in range(0, 366, W):
       """
       find maximum ndvi of this window
       """
       idx = np.ma.argmax(ndvi[day:day+W], axis=0)
       MVC_idxs.append(idx)
       n = np.ma.max(ndvi[day:day+W], axis=0)
       maxNDVI.append(n)
   MVC_idxs = np.array(MVC_idxs)
   maxNDVI = np.array(maxNDVI)
   maxNDVI = np.ma.MaskedArray(data=maxNDVI, mask=np.logical_and(maxNDVI<0, maxNDVI>1))
   """
   Do difference between each ndvi composite
   [description]
   """
   gradNDVI = np.diff(maxNDVI)
   hsdates = np.argwhere(hs).T
   tf = np.zeros((xsize, ysize))
   tf[hsdates[1], hsdates[2]] = hsdates[0]
   """
   Normalise post-fire to prefire
   """
   scarMaps = []
   nT = maxNDVI.shape[0]
   ndiffs = []
   ndviNorms = []
   for day, t in zip(xrange(0, 366, W), xrange(nT-1)):
       # find any hotspots
       hotspots_exist = np.logical_and(tf[tf>0] > day, tf[tf>0] < day + W).sum()
       if hotspots_exist >0:
           """
           Should be some burnt area
           so conduct algorithm
           """
           postfire = maxNDVI[t+1]
           prefire = maxNDVI[t]
           trend = postfire[tf==0].mean() - prefire[tf==0].mean()
           ndviNorm = postfire  - trend
           Nmeant = postfire[tf==0].mean()
           Nmeant_1 = prefire[tf==0].mean()
           Nstdt = postfire[tf==0].std()
           #ndiff = ndviNorm -prefire
           """
           this is wrong from original atm
           """
           ndiff = postfire -prefire
           """
           ^ wrong
           """
           ndiffs.append(ndiff)
           ndviNorms.append(ndviNorm)
           # find where hotspots are
           hs_locs = np.where(np.logical_and(tf>day, tf<day+W))
           # get burnt class
           hs_ndvidiff = ndiff[hs_locs]
           decrease = hs_ndvidiff < 0
           if decrease.sum() ==0 :
               # not proper fires
               continue
           else:
               CBPmean = ndiff[hs_locs][decrease].mean()
               CBPstd = ndiff[hs_locs][decrease].std()
           """
           step 5. Apply a regional NDVI diff threshold
           [description]
           """
           threshold = CBPmean+ gstd*CBPstd
           bb = ndiff < threshold
           """
           apply filters
           """
           mode = ndimage.generic_filter(bb, modeFilter, size = 3)
           """
           get list of CBP again
           """
           CBPs = np.logical_and(ndiff<0, np.logical_and(tf>day, tf<day+W))
           """
           Now segment into separate burnt patches
           """
           import skimage.measure
           scars = skimage.measure.label(mode, neighbors=8)
           """
           now look at each scar and
           1. check it has a CBP
           2. meets threshold test
           """
           for sc in xrange(1, scars.max()):
               scc = scars==sc
               # check it has a CBP

               hasCBP = np.logical_and(CBPs, scc)
               if hasCBP.sum() ==0:
                   # has no CBPs
                   scars[scars==sc]=0
                   continue
               else:
                   # has CBPs
                   """
                   Now compute local stats
                   """
                   clusMean = ndiff[np.logical_and(CBPs, scc)].mean()
                   clusStd = ndiff[np.logical_and(CBPs, scc)].std()
                   """
                   retain only pixels with atleast
                   mean + 1st decrease in NDVI
                   [description]
                   """
                   burnt_cluster = ndiff[scars==sc] < clusMean + gstd * clusStd
                   print gstd * clusStd
                   scars[scars==sc][~burnt_cluster]=0
                   # and obviously decreases
                   decreases = ndiff[scars==sc] < 0
                   scars[scars==sc][~decreases]=0
           """
           Should be finished now
           """
           scars = (scars > 0).astype(float)
           scars *= (day +0.5*W)
           scarMaps.append(scars)
       else:
           """
           No hotspots so no ba?
           """
           empty = np.zeros((xsize, ysize))
           scarMaps.append(empty)
   # to array
   scarMaps = np.array(scarMaps)
   #ndiffs = np.array(ndiffs)
   """
   collapse this to dob
   """
   out = np.zeros((xsize, ysize))
   ddob = np.where(scarMaps>0)
   for t,x, y in zip(ddob[0], ddob[1], ddob[2]):
       out[x,y] = scarMaps[t,x,y]
   """
   now return best estimate of date
   """
   return out


def GEMI(r):
   NIR = r[:, 1]
   RED = r[:, 0]
   gamma = ( 2* (NIR**2-RED**2) + 1.5*NIR + 0.5*RED) /( NIR + RED + 0.5 )
   gemi = gamma * (1 - 0.25 * gamma ) - ( RED - 0.125 )/( 1 - RED   )
   return gemi



def fire_cci(r, hs, nbDistThreshold=39, unburntThresholdBin=1, gemiThreshold=0.9):
   """

   parameters:

   nbDistThreshold: distance from af for a pixel
       to be unburnt (typically 39 at 500m) (int)

   unburntThresholdBin: threshold in cdf. default is 1 (int 0-9)

   gemiThreshold: threshold for annual GEMI comparision
       default is 0.9 (float)

   """
   import datetime
   start_date = datetime.datetime(2008, 1, 1)
   end_date = datetime.datetime(2008, 12, 31)

   doys = [d for d in xrange(1,366) ]
   dates = np.array([datetime.datetime.strptime("2008%03d" %f, "%Y%j") for f in doys])


   xsize = r.shape[2]
   ysize = r.shape[3]

   """
   If there are active fires in this month
   make a composite
   """
   from scipy.spatial import KDTree

   mnths = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
   mnthsj = np.cumsum(mnths)

   day0 = 0

   # annual max GEMI
   MAX_GEMI = GEMI(r)
   MAX_GEMI = MAX_GEMI.max(axis=0)

   mnth_dob = np.zeros((12, xsize, ysize))


   """
   can speed up algorithm
   by loading the nearest neighbour
   search stuff from a file...
   """
   filenn = True
   if filenn:
       dists = np.load("fccidistances.npy")
       nearests = np.load("fccinearests.npy")
   else:
       dists = []
       nearests = []
   ii = 0
   for month in xrange(12):
       day1 = mnthsj[month]
       month_hs = hs[day0:day1]

       # check any hotspots
       if month_hs.sum() >0:

           """
           Yes
               get observation following nearest
               active fire
           """

           ft, fx, fy = np.where(month_hs)
           fire_locs = np.vstack([fx, fy]).T
           vor = cKDTree( fire_locs )

           xy = np.mgrid[0:xsize:1, 0:ysize:1].reshape(2, -1)
           if filenn:
               nn = dists[ii], nearests[ii]
               ii += 1
           else:
               nn = vor.query(xy.T, k=1)
               dists.append(nn[0])
               nearests.append(nn[1])
           #return nn

           dist = nn[0].reshape((xsize, ysize))
           idx = nn[1].reshape((xsize, ysize))

           """
           Now need to find the right observation
           so make day of af for this mask
           """
           TS = ft[nn[1]].reshape((xsize, ysize)) + day0

           """
           Now collect observation
           1. Lowest NIR following this day
               but only up to end of month + 1 month
           """
           """
           for each x y find min after this date
           """
           composite = np.zeros((xsize, ysize))
           red = np.zeros((xsize, ysize))
           mnth_potDOB = np.zeros((xsize, ysize))
           before_ = np.zeros((xsize, ysize))
           for x in xrange(xsize):
               for y in xrange(ysize):
                   string = r[ TS[x, y]:day1+30, 1, x, y]
                   composite[x,y]=np.ma.min(string)
                   idx = np.ma.argmin(string)
                   # and save red for gemi
                   red[x,y] = r[ TS[x, y]:day1+30, 0, x, y][idx]
                   mnth_potDOB[x,y]=idx
                   before_[x,y]=r[ day0-30:TS[x, y], 1, x, y].min()

           """
           Now select burnt CDF from hotspots
           use minimum within window
           """
           burnt_pixels = composite[fx, fy]
           diff = burnt_pixels - before_[fx, fy]
           # limit to smaller quarter
           # mask not decreasing pxls
           decreases = np.logical_and((diff < 0).flatten(), burnt_pixels>0)
           burnt_pixels[decreases]
           burnt_pixels = burnt_pixels[:int(0.25*burnt_pixels.shape[0])]

           """
           make burnt cdf
           """
           burnt_pixels = burnt_pixels[~np.isnan(burnt_pixels)]
           weight, burntbin = np.histogram(burnt_pixels, bins=10,normed=True)
           """
           make unburnt cdf
           """
           dist = nn[0].reshape((xsize, ysize))
           unburnt = np.logical_and(dist > nbDistThreshold, composite>0)
           unburnt_pixels = composite[unburnt]
           unburnt_pixels = unburnt_pixels[~np.isnan(unburnt_pixels)]
           weight, unburntbin= np.histogram(unburnt_pixels, bins=10, normed=True)

           """
           select thresholds
           """
           TB = unburntbin[unburntThresholdBin]
           burnt_threshold = burntbin[(burntbin < TB)][-1]
           # initial
           burnt_ = np.logical_and(composite<burnt_threshold, composite-before_<0)
           burnt_ = np.logical_and(burnt_, dist==0)
           import copy
           new_burnt = np.copy(burnt_)
           """
           Add region growing
           """
           # redefine threshold
           if burntbin[(burntbin < TB)].shape[0]>=7:
               # errr what
               burnt_threshold= burntbin[8]

           preveSum =0
           ee = []
           k = 0
           growing = True
           while growing:
               """
               This will iterate until no new pixels are added
               """
               gg = np.gradient(new_burnt)
               edges = gg[0]*gg[1]>0
               #edges = np.logical_not(~edges, burnt_)
               # test these pixels
               ee.append(edges)
               condition1 = composite[edges]<burnt_threshold
               # ambigious decline in NIR should be larger than for unburnt areas
               d = composite[unburnt] - before_[unburnt]
               condition2 = composite[edges] - before_[edges] < d[(d<0)].mean()
               """
               gemi thingy
               """
               mnth_gemi = GEMI(np.stack((red, composite), axis=1))
               diffGEMI = mnth_gemi - MAX_GEMI
               # burnt pixel GEMI
               logic = np.logical_and(diffGEMI[burnt_]<0, diffGEMI[burnt_]>-1)
               threshold = diffGEMI[burnt_][logic].mean()
               conditionGEMI = gemiThreshold * diffGEMI[edges] < threshold

               """
               Now label the edge pixels that must be burnt
               """
               burnt_edges = np.logical_and(condition1, np.logical_and(condition2, conditionGEMI))
               # make these pixels burnt
               grow = np.zeros((xsize, ysize)).astype(bool)
               grow[edges]=burnt_edges
               new_burnt =np.logical_or(grow , new_burnt)

               identified_k = burnt_edges.sum()
               eSum = edges.sum()
               if identified_k>0 and eSum!=preveSum:
                   # we grew a bit
                   growing = True
                   preveSum = eSum
               else:
                   growing = False
               k+=1

               # still not sure about this...
           """
           This is the final classification
           """
           mnth_dob[month][new_burnt]=mnth_potDOB[new_burnt]
       # move forward day0
       day0 = day1

   """
   to make comparision codes
   easier we collapse to 1d array
   """
   out = np.zeros((xsize, ysize))
   ddob = np.where(mnth_dob>0)
   for t,x, y in zip(ddob[0], ddob[1], ddob[2]):
       out[x,y] = mnth_dob[t,x,y]
   return out



def L3JRC(refl, hs, gstd=2, b2threshold=0.26, b6threshold=0.25):
   """
   parameters:

       gstd: number of standard deviations below mean
           to be burnt (default 2) (float)


       b2threshold: band2 (nir) threshold (default 0.26) (float)

       b6threshold: band6 (swir) threshold (default 0.25) (float)

   """
   xs, ys = refl.shape[-2], refl.shape[-1]
   nir = refl[:, 1] #np.ma.array(refl[:, 1], mask=~qa)
   b6 = refl[:, 5] #np.ma.array(refl[:, 5], mask=~qa)

   IC_nir = np.zeros_like((nir))
   I = np.zeros((nir.shape[0]))
   #rel_idx = np.where( qa )[0]
   idx_acc = 0
   for doy in xrange(366):
       nir_vals = nir[:doy]

       IC_nir_i = np.ma.mean(nir_vals, axis=0)
       IC_nir[doy] = IC_nir_i
       idx_acc  += 1

   metric =  (nir-IC_nir)/(nir+IC_nir)
   """
   Get mean and standard deviation for I
   over the 200x200 window
   """

   # mask nans
   #new_mask = np.logical_and(metric.mask, np.isnan(metric))
   #metric.mask = new_mask
   #meanI = np.nanmean(metric, axis=0)
   #stdI = np.nanstd(metric, axis=0)

   meanI  = np.nanmean(metric)
   stdI = np.nanstd(metric)

   """
   A pixel is burnt if any I
   falls 2std below the mean
   """
   burns = metric < (meanI-gstd*stdI)
   """
   threshold checks
    830 nm (S1 pixel value < 260)
    and 1660 nm (S1 pixel value > 250)
   -- equiv modis bands are
   band 2 (1)
   band 6 (5)
   """
   band2Thres = nir < b2threshold
   band5Thres = b6 > b6threshold
   thresh = np.logical_and(band2Thres, band5Thres)
   burns = burns & thresh
   dob, x, y = np.where(burns)
   out = np.zeros((xs,ys))
   out[x,y] = dob
   return out
