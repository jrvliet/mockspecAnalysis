
'''
Code that allows for custom TPCF samples.
Mixes samples from muptiple galaxies, inclinations, azimuthal angles
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import itertools as it
import joblib as jl
import numba as nb
import tempfile
import sys
import decimal
import os
import time


class tpcfRun(object):
    '''
    Class that holds the settings for this interactive run
    '''
    
    def __init__ (self):
        self.azLo = 0.
        self.azHi = 90.
        self.expn = '0.490'
        self.ions = 'MgII CIV OVI'.split()
        self.iLo = 0.
        self.iHi = 90.
        self.dLo = 0.
        self.dHi = 200.
        self.loc = '/mnt/cluster/abs/cgm/vela2b/'
        self.runBoot = 0
        self.binSize = 10.
        self.bootNum = 1000
        self.ncores = 4
    
        # Time selection
        self.zRange = 0
        self.loZ = '1.05'
        self.hiZ = '1.00'
        self.loA = 1./float(self.loZ)-1.
        self.hiA = 1./float(self.hiZ)-1.

        # Mass selection
        self.useMass = 0
        self.massType = 1
        self.loMass = 10
        self.hiMass = 12

        # SFR selection
        self.useSFR = 0
        self.sfrType = 1
        self.loSFR = -12
        self.hiSFR = -9

        # Filename 
        self.outName = 'tpcfTesting.h5'

    def print_run(self):
        s = ''
        s += '\nTPCF Run Properties: \n'
        s += '\tLocation = {0:s}\n'.format(self.loc)

        s += '\tRedshift Selection\n'
        if self.zRange==0:
            s += '\t\tSingle snapshot at {0:s}\n'.format(self.loZ)
        else:
            s += '\t\tRange from {0:s} to {1:s}\n'.format(self.loZ,self.hiZ)

        s += '\tMass Selection\n'
        if self.useMass==0:
            s += '\t\tNo mass selection applied\n'
        else:
            if self.massType==1:
                s += '\t\tHalo mass range = {0:.1f} - {1:.1f}\n'.format(
                            self.loMass,self.hiMass)
            else:
                s += '\t\tStellar mass range = {0:.1f} - {1:.1f}\n'.format(
                            self.loMass,self.hiMass)
                
        s += '\tStar Formation Rate Selection\n'
        if self.useSFR==0:
            s += '\t\tNo SFR selection applied\n'
        else:
            if self.sfrType==1:
                s += '\t\tSpecfic SFR range = {0:.1f} - {1:.1f}\n'.format(
                        self.loSFR,self.hiSFR)
            else:
                s += '\t\tSFR range = {0:.1f} - {1:.1f}\n'.format(
                        self.loSFR,self.hiSFR)
        
        s += '\tLOS Selection:\n'
        s += '\t\tAzimuthal Limits = {0:f} - {1:f}\n'.format(self.azLo,self.azHi)
        s += '\t\tImpact Limits = {0:f} - {1:f}\n'.format(self.dLo,self.dHi)
        s += '\t\tInclination Limits = {0:f} - {1:f}\n'.format(self.iLo,self.iHi)

        s += '\tOutput file = {0:s}\n'.format(self.outName)
    
        s += '\tIons = {0:s}\n'.format(', '.join(self.ions))
        
        return s

def control(run):
    
    '''
    Main function for the interactive TPCF
    Controls everything
    '''

    selections = galaxy_selection(run)

    los = select_los(run,selections)
    allVelsPath,allVelsShapes,maxVel = build_sample(run,los)

    bins,labels = sample_bins(run,maxVel)#,tpcfProp)
    tpcfs = sample_tpcf(run,allVelsPath,allVelsShapes,bins,labels)
    
    print(allVelsShapes)
    print(type(tpcfs),len(tpcfs),len(bins),len(labels))
    for t in tpcfs:
        print(len(t))
    #sys.exit()
    if run.runBoot == 1:
        print('Begin bootstrap')
        means,stds = bootstrap(run,allVelsPath,allVelsShapes,bins,labels)
        print('Means shape = ',means.shape)
        print('Std shape = ',stds.shape)

    # Put full TPCFs into dataframe
    tpcfFull = pd.DataFrame(index=labels)
    tpcfTypes = 'full mean std'.split()
    cols = pd.MultiIndex.from_product([run.ions,tpcfTypes])
    tpcfFull = pd.DataFrame(index=labels,columns=cols)
    for i,(ion,tpcf) in enumerate(zip(run.ions,tpcfs)):
        # Pad the array with nans
        padWidth = len(bins)-len(tpcf)
        print(ion,padWidth,len(bins),len(tpcf),len(means[:,i]))
        if padWidth>0:
            tpcf = np.pad(tpcf,(0,padWidth),mode='constant',
                            constant_values= (np.nan))
            if run.runBoot==1:
                mean = np.pad(means[:,i],(0,padWidth),mode='constant',
                                constant_values= (np.nan))
                std = np.pad(stds[:,i],(0,padWidth),mode='constant',
                                constant_values= (np.nan))
        elif padWidth<0:
            tpcf = tpcf[:len(bins)]
            if run.runBoot==1:
                mean = means[:len(bins),i]
                std = stds[:len(bins),i]
        
        tpcfFull[ion,'full'] = tpcf
        if run.runBoot==1:
            print(tpcfFull.shape,mean.shape)
            tpcfFull[ion,'mean'] = mean
            tpcfFull[ion,'std'] = std
        


    #header = 'vel '+' '.join(run.ions)
    #header = header.split()
    #tpcfFull.columns = header
    tpcfFull.index.name = 'vel'

    write_tpcf(tpcfFull,run)

    cleanup(allVelsPath)

    print('Done')
    
        
def galaxy_selection(run):
    '''
    Reads in the sfr files and returns all galaxies and expansion
    paramters within the mass, sfr, and expn limits
    '''

    sfrLoc = run.loc+'sfr/'
    filename = 'vela2b-{0:d}_sfr.csv'

    galNums = range(21,30)
    selection = []
    for galNum in galNums:
        fname = sfrLoc+filename.format(galNum)
        df = pd.read_csv(fname)
        if run.zRange==1:
            timeSelection = (df['a']>=run.loA) & (df['a']<=run.hiA)
        else:
            timeSelection = np.islcos(df['a'],0.22)

        if run.useMass==1:
            if run.massType==1:
                massSelection = ((df['mvir']>=run.loMass) & 
                                 (df['mvir']<=run.hiMass))
            else:
                massSelection = ((df['mstar']>=run.loMass) & 
                                 (df['mstar']<=run.hiMass))

        if run.useSFR==1:
            if run.sfrType==1:
                sfrSelection = ((df['ssfr']>=run.loSFR) & 
                                (df['ssfr']<=run.hiSFR))
            else:
                sfrSelection = ((df['sfr']>=run.loSFR) & 
                                (df['sfr']<=run.hiSFR))


        fullSelection = df[timeSelection & sfrSelection & massSelection]['a']
        numSnaps = len(fullSelection)
        if numSnaps==1:
            print('Halo {0:d} - Using {1:d} snapshot'.format(galNum,numSnaps))
        else:
            print('Halo {0:d} - Using {1:d} snapshots'.format(galNum,numSnaps))

        
        for a in fullSelection:
            selection.append((galNum,'{0:.3f}'.format(a)))
        
    return selection




def read_input():
    '''
    Reads in the input file and fills out run object
    '''
    
    fname = 'tpcf.config'
    run = tpcfRun()
    with open(fname,'r') as f:
        # Read in time section
        for i in range(2):
            f.readline()
        run.zRange = int(f.readline().split()[0])
        run.loZ = f.readline().split()[0]
        run.hiZ = f.readline().split()[0]
        run.loA = 1./(float(run.loZ)+1)
        run.hiA = 1./(float(run.hiZ)+1)

        # Read in the mass section
        for i in range(2):
            f.readline()
        run.useMass = int(f.readline().split()[0])
        run.massType = int(f.readline().split()[0])
        run.loMass = float(f.readline().split()[0])
        run.hiMass = float(f.readline().split()[0])

        # Read in star formation rate section
        for i in range(2):
            f.readline()
        run.useSFR = int(f.readline().split()[0])
        run.sfrType = int(f.readline().split()[0])
        run.loSFR = float(f.readline().split()[0])
        run.hiSFR = float(f.readline().split()[0])
    
        # Read in LOS section
        for i in range(2):
            f.readline()
        run.azLo = float(f.readline().split()[0])
        run.azHi = float(f.readline().split()[0])
        run.dLo = float(f.readline().split()[0])
        run.dHi = float(f.readline().split()[0])
        run.iLo = float(f.readline().split()[0])
        run.iHi = float(f.readline().split()[0])

        # Read in TPCF Settings
        for i in range(2):
            f.readline()
        run.binSize = int(f.readline().split()[0])
        run.runBoot = int(f.readline().split()[0])
        run.bootNum = int(f.readline().split()[0])
        run.ncores = int(f.readline().split()[0])

        # Read in the output filename
        for i in range(2):
            f.readline()
        run.outName = f.readline().split()[0]+'.h5'

        # Read in ions
        for i in range(2):
            f.readline()
        ions = []
        for line in f:
            ions.append(line.split()[0])
        run.ions = ions
            
    return run
        

def select_los(run,selections):
    '''
    Selects lines of sight that fit the limits contained in run
    '''

    linesHeader = 'los impact phi incline az'.split()
    galNums = range(21,30)
    iDirs,iDirsList = find_inclinations(run,selections)
        
    print('iDirs: ',iDirs)
    los = pd.DataFrame(columns=pd.MultiIndex.from_tuples(iDirsList),
                        index=range(1000))
    
    # Loop through to read in lines.info files
    for gal,inc in iDirsList:
        dirname = '{0:s}/vela{1:d}/a{2:s}/{3:s}/'.format(run.loc,
                                                gal,run.expn,inc)
        linesFile = '{0:s}/lines.info'.format(dirname)
        lines =  pd.read_csv(linesFile,names=linesHeader,
                             sep='\s+',skiprows=1)
        targets = ((lines['az']>=run.azLo) & 
                   (lines['az']<=run.azHi) &
                   (lines['impact']>=run.dLo) & 
                   (lines['impact']<=run.dHi))
        target = lines['los'][targets]
        
        los[gal,inc] = pd.Series(target)
        los.set_value(inc,gal,target)
        
    return los
   
def build_sample(run,los):
    '''
    Using LOS contained in los, select out the velocity differences 
    from TPCF files
    '''
    
    flog = open('tpcf_errors.log','w')
    flog.write('GalNum\tExpn\tIncline\tIon\n')
    flogs = '{0:d}\t{1:s}\t{2:s}\t{3:s}\n'

    # Loop through columns in los
    allVels = [pd.DataFrame() for ion in run.ions]
    for galNum,inc in los.columns:
    
        dirname = '{0:s}/vela{1:d}/a{2:s}/{3:s}/tpcf/'.format(run.loc,
                    galNum,run.expn,inc)
        for i,ion in enumerate(run.ions):
            filename = 'vela2b-{0:d}_{1:s}_{2:s}_{3:s}_velDiff.csv'.format(
                    galNum,run.expn,inc,ion)
            fname = dirname+filename
            try:
                vd = pd.read_csv(fname)
            except IOError:
                flog.write(flogs.format(galNum,run.expn,inc,ion))
                continue
            velDiffColumns = set(vd.columns.values)
            
            # Select out the lines that are in LOS
            snap = los[galNum,inc]
            a = set(['{0:.1f}'.format(v) for v in snap.values])
            snapVels = vd[list(a & velDiffColumns)]
            allVels[i] = pd.concat([allVels[i],snapVels],axis=1)
    
    # Reset allVels column names
    for df in allVels:
        df.columns = range(df.shape[1])

    # Convert to memmap objects
    allVelsPaths = []
    allVelsShapes = []
    maxVel = 0
    for df,ion in zip(allVels,run.ions):
        #path = tempfile.mkdtemp()
        path = os.path.join('.','tmp')
        
        velMemPath = os.path.join(path,
                    'vellDiff_{0:s}.mmap'.format(ion))
        velDiffMem = np.memmap(velMemPath,dtype='float',
                        shape=df.shape,mode='w+')
        velDiffMem[:] = df.values[:]
        allVelsPaths.append(velMemPath)
        allVelsShapes.append(df.shape)
        dfMax = np.nanmax(df.values)
        if dfMax>maxVel:
            maxVel = dfMax

    return allVelsPaths,allVelsShapes,maxVel
    

def sample_bins(run,maxVel):#,tpcfProp):
    ''' 
    Generates the velocity bins and labels to making the tpcf
    '''

    nbins = int(maxVel/run.binSize)
    endPoint = run.binSize*(nbins+1)
    bins = np.arange(0,endPoint,run.binSize)

    labels = [(bins[i]+bins[i+1])/2. for i in range(nbins)]
    lastLabel = labels[-1] + (labels[1]-labels[0])
    labels.append(lastLabel)
    return bins,labels
    


def sample_tpcf(run,samplePaths,sampleShapes,bins,labels,resample=0):
    '''
    Constructs the TPCF from the sample
    '''
    
    tpcfs = []
    for sPath,sShape in zip(samplePaths,sampleShapes):
        print(sPath)
        print(sShape)
        print(len(bins),len(labels))
        sample = np.memmap(sPath,dtype='float',mode='r',
                            shape=sShape)
        if resample!=0:
            sample = sample[:,np.random.choice(sample.shape[1],
                            sample.shape[1],replace=True)]
        
        flat = sample.flatten()
        print(flat.shape)
        flat = flat[~np.isnan(flat)]
        #tpcf = np.sort(np.bincount(np.digitize(flat,bins),minlength=len(bins)))[::-1]
        #tpcf = tpcf/tpcf.sum()
        tpcf,bin_edges = np.histogram(flat,bins,density=True)
        print(len(tpcf))
        tpcfs.append(tpcf)
    return tpcfs
    

def bootstrap_tpcf(run,sample,shape,bins,labels):
    '''
    Constructs the TPCF during bootstraps
    '''

    # Resample the data set with replacement
    print('Start resample',flush=True)
    sample = sample[:,np.random.choice(sample.shape[1],
                    sample.shape[1],replace=True)]
    flat = sample.flatten()
    flat = flat[~np.isnan(flat)]
    tpcf,bin_edges = np.histogram(flat,bins,density=True)
    #tpcf = np.sort(np.bincount(np.digitize(flat,bins)))[::-1]
    tpcf = tpcf/tpcf.sum()
    print('End resample',flush=True)
    
    return tpcf

    
def bootstrap(run,allVelsPaths,allVelsShapes,bins,labels):

    '''
    Performs bootstraps for uncertainty analysis
    Returns mean and std dev tpcfs
    '''

    mempath = os.path.join('./tmp/','bootstrap.mmap')
    bootArr = np.memmap(mempath,dtype='float',
                        shape=(len(bins)+1,run.bootNum),mode='w+')
    bootMean = np.zeros((len(bins)+1,len(run.ions)))
    bootStd = np.zeros_like(bootMean)

    for i,(ion,path,shape) in enumerate(
                                zip(run.ions,allVelsPaths,allVelsShapes)):
        print('\n{0:s}'.format(ion))
        sample = np.memmap(path,dtype='float',mode='r',shape=shape)
        print('Read in {0:s} with shape = '.format(path),sample.shape)
        run.ncores=1
        #jl.Parallel(n_jobs=run.ncores,verbose=9)(
        #    jl.delayed(bstrap)(run,sample,shape,bins,labels,bootArr,j)
        #    for j in range(run.bootNum))
    
        for j in range(run.bootNum):
            #bstrap(run,sample,shape,bins,labels,bootArr,j)        
            bstrap(run,path,shape,bins,labels,bootArr,j)        
    
        print('Finished bootstrap for {0:s}'.format(ion))
        bootMean[:,i] = np.nanmean(bootArr,axis=1)
        bootStd[:,i] = np.nanstd(bootArr,axis=1)
        
    return bootMean,bootStd
        

#@nb.jit
def bstrap(run,path,shape,bins,labels,bootArr,i):
#def bstrap(run,sample,shape,bins,labels,bootArr,i):

    sample = np.memmap(path,dtype='float',mode='r',shape=shape)
    print('\n',i,flush=True)
    print('Start resample',flush=True)
    start = time.time()
    bootCols = np.random.choice(sample.shape[1],sample.shape[1],replace=True)
    print('Time to select columns = ',time.time()-start)
    #sample = sample[:,bootCols]
    sample = np.take(sample,bootCols,axis=1)
    print('Time to make new sample = ',time.time()-start)
    print('\tNew Sample Made',flush=True)
    flat = sample.flatten()
    print('\tFlattened',flush=True)
    flat = flat[~np.isnan(flat)]
    print('\tRemoved NaNs',flush=True)
    #tpcf = np.sort(np.bincount(np.digitize(flat,bins)))[::-1]
    tpcf,bin_edges = np.histogram(flat,bins,density=True)
    print('\tTPCF made',flush=True)
    tpcf = tpcf/tpcf.sum()
    print('\tTPCF Normalized',flush=True)
    bootArr[:len(tpcf),i] = tpcf
    print('End resample',flush=True)
    #resample = 1
    #tpcf = bootstrap_tpcf(run,sample,shape,bins,labels)

def find_inclinations(run,selections):

    '''
    Returns a list of inclinations directories for each galaxy number
    '''

    iDirs = {}
    
    for galNum,expn in selections:

        # Check if the expansion parameter exists
        subloc = run.loc+'vela{0:d}/a{1:s}/'.format(galNum,expn)
        dirname = os.path.join(run.loc,subloc)
        inclines = []
        if os.path.isdir(dirname):
            
            # Get list of inclinations in this directory
            inclines = [name for name in os.listdir(dirname) if 
                        os.path.isdir(os.path.join(dirname,name)) and
                        name[0]=='i' and 
                        float(name.split('i')[1])>=run.iLo and
                        float(name.split('i')[1])<=run.iHi]

        iDirs[galNum] = inclines

    iDirsList = []
    for key,val in iDirs.items():
        for v in val:
            iDirsList.append([key,v])
    
    return iDirs,iDirsList
    

def cleanup(paths):

    '''
    Deletes mmaps
    '''
    command = 'rm {0:s}'
    for path in paths:
        sp.call(command.format(path),shell=True)


def write_tpcf(df,run):
    '''
    Writes the full tpcf to file with run metadata
    '''

    with pd.HDFStore(run.outName) as store:
        store.put('data',df)
        store.get_storer('data').attrs.metadata = run.__dict__
        
    
        

if __name__ == '__main__':

    run = read_input()
    #run.loc = '/home/sims/vela2b/'
    print(run.print_run())


    control(run)

