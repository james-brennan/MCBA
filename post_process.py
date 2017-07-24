"""
post_process.py

Put files together into one file
""" 
import glob 
import numpy as np

for prod in ['fcci', 'mcd64', 'l3jrc', 'hands']:
    f = glob.glob("/home/users/jbrennan01/DATA2/MCBA/output/%s/%s*npz" % (prod, prod))
    # figure out keys
    ftp = np.load(f[0]).keys()
    # setup
    fileout =  {}
    for key in ftp:
        fileout[key]=[] 
    # load and dump to dict
    for file in f:
        a = np.load(file)
        for key in ftp:
            fileout[key].append(a[key])
            print file, key
    # make into arrays
    for key in fileout:
        fileout[key] = np.array(fileout[key])
    # save
    np.savez("/home/users/jbrennan01/DATA2/MCBA/Traces_%s.npz" % prod, **fileout)

