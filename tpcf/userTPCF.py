
'''
Second attempt at interactive TPCF
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

def control(run):

    '''
    Main control function
    '''

    # Select out galaxies
    print('\nSelect Galaxies')
    selections = galaxy_selection(run)
    
    # Select out LOS
    print('\nSelect LOS')
    los = select_los(run,selections)

    # Read in all velDiffs from mockspec's TPCFs
    # Store in mmap objects
    print('\nBuild Sample')
    velPaths,velShapes,velMax = build_sample(run,los)
    
    # Create the bins and velocity labels
    print('\nCreate Bins')
    bins,labels = sample_bins(run,velMax)

    # Create dataframe to hold results
    print('\nCreate dataframe')
    columns = [run.ions,'Full Mean Std'.split()]
    columns = pd.MultiIndex.from_product(columns, names=['Ions', 'Fields'])
    full = pd.DataFrame(index=labels,columns=columns)
    full.index.name = 'Velocity'

    # Loop over ions
    for ion,velPath,velShape in zip(run.ions,velPaths,velShapes):

        print('\nIon = {0:s}'.format(ion))
        # Make the full TPCF for this ion
        tpcf = sample_tpcf(velPath,velShape,bins)
        print('Size of TPCF = {0:d}'.format(len(tpcf)))

        # Bootstrap if needed
        if run.runBoot==1:
            print('Starting bootstrap')
            m,s = bootstrap(run,velPath,velShape,bins)
        else:
            print('Skipping bootstrap')
        
        # Build a dataframe to hold the results
        
        full[ion,'Full'] = np.pad(tpcf,(0,len(bins)-len(tpcf)),'constant')
        full[ion,'Mean'] = np.pad(m,(0,len(bins)-len(m)),'constant')
        full[ion,'Std'] = np.pad(s,(0,len(bins)-len(s)),'constant')
        

    # Save to disk
    print('\nSaving...')
    #full.to_csv(run.outName)
    with pd.HDFStore(run.outName) as store:
        store.put('data',full)
        store.get_storer('data').attrs.metadata = run.__dict__
    print('\nDone')

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
        self.ewLo = 0.
        self.ewHi = 10.
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
        run.ewLo = float(f.readline().split()[0])
        run.ewHi = float(f.readline().split()[0])

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
                    'velDiff_{0:s}.mmap'.format(ion))
        velDiffMem = np.memmap(velMemPath,dtype='float',
                        shape=df.shape,mode='w+')
        velDiffMem[:] = df.values[:]
        allVelsPaths.append(velMemPath)
        allVelsShapes.append(df.shape)
        dfMax = np.nanmax(df.values)
        if dfMax>maxVel:
            maxVel = dfMax

    return allVelsPaths,allVelsShapes,maxVel


def sample_bins(run,maxVel):
    ''' 
    Generates the velocity bins and labels to making the tpcf
    '''

    nbins = int(np.ceil(maxVel/run.binSize))
    endPoint = run.binSize*(nbins+1)
    bins = np.arange(0,endPoint,run.binSize)

    labels = [(bins[i]+bins[i+1])/2. for i in range(nbins)]
    lastLabel = labels[-1] + (labels[1]-labels[0])
    labels.append(lastLabel)
    return bins,labels

def sample_tpcf(velPath,velShape,bins,resample=0):
    '''
    Builds a TPCF from the data
    '''

    # Read in data
    sample = np.memmap(velPath,dtype='float',mode='r',shape=velShape)
    
    # Resample if part of bootstrap
    if resample==1:
        #print('\tStart resample')
        newCols = np.random.choice(velShape[1],velShape[1],replace=True)
        sample = np.take(sample,newCols,axis=1)
        #sample = sample[:,np.random.choice(velShape[1],velShape[1],replace=True)]
        #print('\tEnd resample')

    # Flatten data (LOS number doesn't matter)
    sample = sample.flatten()
    
    # Remove NaNs
    sample = sample[~np.isnan(sample)]
    
    # Bin results and normalize
    #tpcf = np.sort(np.bincount(np.digitize(sample,bins)))[::-1]
    tpcf = np.bincount(np.digitize(sample,bins))
    tpcf = tpcf/tpcf.sum()
    #tpcf,bin_edges = np.histogram(sample,bins,density=True)
    
    return tpcf
    

def bootstrap(run,velPath,velShape,bins):
    '''
    Controls bootstrapping
    Returns mean and std
    '''
    
    # Create mmap to hold all the bootstrap TPCFs
    bootPath = './tmp/bootstrap.mmap'
    boot = np.memmap(bootPath,dtype='float',
                    shape=(run.bootNum,len(bins)),mode='w+')
    
    print('\nCreated boot array')
    jl.Parallel(n_jobs=run.ncores,verbose=5)(
        jl.delayed(bstrap)(velPath,velShape,bins,boot,i)
        for i in range(run.bootNum))

    m = np.nanmean(boot,axis=0)
    s = np.nanstd(boot,axis=0)
        
    return m,s
    
    

def bstrap(velPath,velShape,bins,boot,i):
    tpcf = sample_tpcf(velPath,velShape,bins,1)
    #print('In bstrap i={0:d}, len(tpcf) = {1:d}, len(bins) = {2:d}'.format(i,
    #            len(tpcf),len(bins)))
    boot[i,:len(tpcf)] = tpcf
    


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

