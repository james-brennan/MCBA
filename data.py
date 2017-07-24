"""
data.py

This script loads and generates realistic
realisations of the data

"""
import h5py
import numpy as np
import scipy.stats

def _load_truth():
    b = h5py.File("/group_workspaces/cems2/nceo_generic/users/jbrennan01/RRLandsat/inputDS/ds_h24v03_2008.hdf5", 'r')
    hs = b['terra_hs'][:]
    hs = hs>7
    a = h5py.File("/group_workspaces/cems2/nceo_generic/users/jbrennan01/RRLandsat/models/ref_h24v03_2008.hdf5", 'r')
    iso = a['isotropic'][90:-89]
    #geo = a['geometric'][90:-89]
    #vol = a['volumetric'][90:-89]
    dob = a['DOB'][:]
    return iso, hs, dob


def simpleExperiment(C=np.ones(7), PsPs=0.5, PcPc=0.5):
    """
    This produces data realisations
    with missing observations and gaussian
    noise.

    The starting point for the noise is
    the MODIS sdr error characterisation.
    The values of this are adapted by the constants
    C = [c_1...c_n]. Generally it is easiest to
    assume no covariance between C...


    The model here for missing observations is a simple
    markov chain model for cloudiness. It has two
    parameters,

        PsPs -- probability a sunny day follows a sunny day
        PcPc -- probability a cloudy day follows a cloudy day

    The probability for any day being cloudy is then
    realised through the markov model:

    Pcl(t+1) = W Pcl(t)

    where W is the transition matrix
    """

    """
    * -- Load the data --*
    """
    iso, hs, truedob = _load_truth()

    """
    *-- cloudiness model --*
    """
    PsPc = 1 - PsPs
    PcPs = 1 - PcPc
    """
    transition matrix -- don't really need...
    """
    P = np.array([[PsPs, PsPc],
                  [PcPs, PcPc]])
    """
    initial condintion
    """
    sunny = True
    t0 = np.random.choice([False, True])
    """
    run realisation
    """
    clear = []
    t = t0
    for k in xrange(366):
        # predict tomorrow
        if t == sunny:
            t1 = bool(scipy.stats.bernoulli.rvs(PsPs))
        else:
            t1 = not bool(scipy.stats.bernoulli.rvs(PcPc))
        clear.append(t1)
        t = t1
    clear = np.array(clear)


    """
    *-- Add noise --*

    Model here is that noise is Gaussian and independently
    drawn per pixel per day per waveband... a BIG assumption

    The MODIS sdr characterisation is used as a central value.

    to keep the memory requirement down we do this only along
    time in a loop.

    This may take awhile...
    """
    err = np.array([0.004, 0.015, 0.003, 0.004, 0.013, 0.010, 0.006])
    xsize = 200
    ysize = 300
    noise_ = np.zeros((7, 366))
    for x in xrange(xsize):
        for y in xrange(ysize):
            for band in xrange(7):
                bnoise_ = np.random.normal(0, C[band]*err[band], size=(366))
                noise_[band] = bnoise_
            """
            apply noise to data
            """
            iso[:, :, x, y] += noise_.T



    """
    Apply the qa to the data
    """


    qa = np.zeros((366, 7, 200, 300)).astype(bool)

    for x in xrange(xsize):
        for y in xrange(ysize):
            qa[:, :, x, y]=clear[:, None]
    """
    also mask where water etc
    """
    waterMask = iso !=-999
    qa = np.logical_and(qa,  waterMask)
    iso = np.ma.MaskedArray(data=iso, mask=~qa)

    """
    Done now return the observations, hotspots and
    qa field
    """
    return iso, clear, hs

