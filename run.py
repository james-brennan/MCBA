"""
run.py
This script performs the main task of
1. generating observations
2. choosing parameters
3. running algorithms and saving outputs
"""
import sys
from algorithms import *
import numpy as np
from data import *
import scipy.stats

if __name__ == "__main__":
    job_id = int(sys.argv[1])
    model_run_id = int(sys.argv[2])


    """
    *-- generate observations --*
    for this we need to paramterise the fwd model
    with:
        C -- vector of noise scalars
        PsPs -- probability a sunny day follows a sunny day
        PcPc -- probability a cloudy day follows a cloudy day
    We need to choose some reliable prior distributions
    for these
    C -- we expect marginal variability relative to
        the modis error characterisation...
        It seems that higher values are more likley
        than lower values...
    PsPs and PcPc
    There are many options for these.. Perhaps uniform is
    sensible but never edges... I think a more realistic
    distribution could come from the true MODIS qa
    flag counts?
    but for now lets make gaussian around some mean values
         u    sd
    PsPs 0.6 0.3
    PcPc 0.2 0.2
    OR uniform beta?
    """
    # below goes negative
    #C = np.random.multivariate_normal(np.ones(7), 0.5*np.eye(7))
    # check not negative

    C = scipy.stats.gamma.rvs(1,0, size=7)
    # this has a mean of 1...

    #PsPs = np.random.normal(0.6, 0.3)
    #PcPc = np.random.normal(0.2, 0.2)

    # or maybe beta uniform distribution for non-informative
    # prior?

    # also these are definitely not indepdent!!!!
    # need to think of an importance sampling approach
    PsPs = scipy.stats.beta.rvs(1.0, 1.0 )
    PcPc = scipy.stats.beta.rvs(1.0, 1.0 )

    iso, qa, hs = simpleExperiment(C=C, PsPs=PsPs, PcPc=PcPc)

    """
    *-- Choose parameters for algorithms --*
        These are broadly defined with uniform
        distributions over reliable values of
        the algorithms parameters.
        For documentation of the meaning of each
        parameter consult algorithms.py
    """

    MCD64_params = {
    'W':np.random.randint(4,45),
    'stdSmaxr':np.random.randint(1,15),
    'dBThreshold':np.random.randint(1,10),
    'sigmap':np.random.randint(1,10),
    }

    fcci_params = {
    'nbDistThreshold':np.random.randint(10,90),
    'unburntThresholdBin':np.random.randint(0,9),
    'gemiThreshold':np.random.random()
    }

    hands_params = {
    'W':np.random.randint(10,90),
    'gstd':6*np.random.random(),
    }

    l3_params = {
    'b2threshold':0.8*np.random.random(),
    'b6threshold':0.8*np.random.random(),
    'gstd':6*np.random.random(),
    }

    """
    run algorithms with these parameter settings
    and input data
    """
    bmcd64 = MCD64(iso, hs, **MCD64_params)
    bhands = HANDS(iso, hs, **hands_params)
    bl3 = L3JRC(iso, hs, **l3_params)
    bfcci = fire_cci(iso, hs, **fcci_params)

    """
    Save outputs
    """
    np.savez_compressed("/home/users/jbrennan01/DATA2/MCBA/output/mcd64/mcd64_%i_%i.npz" % (job_id, model_run_id),
            dob=bmcd64, C=C, qa=qa, PsPs=PsPs, PcPc=PcPc, **MCD64_params)
    np.savez_compressed("/home/users/jbrennan01/DATA2/MCBA/output/fcci/fcci_%i_%i.npz" % (job_id, model_run_id),
            dob=bfcci,  C=C, qa=qa, PsPs=PsPs,  PcPc=PcPc,**fcci_params)
    np.savez_compressed("/home/users/jbrennan01/DATA2/MCBA/output/l3jrc/l3jrc_%i_%i.npz" % (job_id, model_run_id),
            dob=bl3,  C=C, qa=qa, PsPs=PsPs,  PcPc=PcPc,**l3_params)
    np.savez_compressed("/home/users/jbrennan01/DATA2/MCBA/output/hands/hands_%i_%i.npz" % (job_id, model_run_id),
            dob=bhands, C=C, qa=qa,PsPs=PsPs, PcPc=PcPc,  **hands_params)

