"""
run.py


This script performs the main task of
1. generating observations
2. choosing parameters
3. running algorithms and saving outputs

"""
import sys
import algorithms
import numpy as np

if __name__ == "__main__":

    model_run_id = sys.argv[1]


    """
    generate observations
    """
    C = np.ones(7)
    iso, qa, hs = simpleExperiment()
    """
    *-- Choose parameters --*

        These are broadly defined with uniform
        distributions over reliable values of
        the algorithms parameters.
        For documentation of the meaning of each
        parameter consult algorithms.py
    """

    MCD64_params = {
    'W':np.random.randint(4,45),
    'Smax':np.random.randint(1,8),
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
    bmcd64 = MCD64(iso_in, hs, **MCD64_params)
    bhands = HANDS(iso_in, hs, **hands_params)
    bl3 = L3JRC(iso_in, hs, **l3_params)
    bfcci = fire_cci(iso_in, hs, **fcci_params)

    """
    Save outputs
    """
    np.savez("/home/users/jbrennan01/DATA2/MCBA/output/mcd64/mcd64_%i.npz" % model_run_id,
            dob=bmcd64, C=C, qa=qa, **MCD64_params)
    np.savez("/home/users/jbrennan01/DATA2/MCBA/output/fcci/fcci_%i.npz" % model_run_id,
            dob=bfcci,  C=C, qa=qa, **fcci_params)
    np.savez("/home/users/jbrennan01/DATA2/MCBA/output/l3jrc/l3jrc_%i.npz" % model_run_id,
            dob=bl3,  C=C, qa=qa, **l3_params)
    np.savez("/home/users/jbrennan01/DATA2/MCBA/output/hands/hands_%i.npz" % model_run_id,
            dob=bhands, C=C, qa=qa, **hands_params)
