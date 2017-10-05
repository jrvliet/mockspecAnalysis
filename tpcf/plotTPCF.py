
import pandas as pd
import matplotlib.pyplot as plt


fname = 'tpcfTesting.h5'
df = pd.read_hdf(fname,'data')

print(df.head())
ions = 'MgII CIV OVI'.split()
fields = 'full mean std'.split()

fig,axes = plt.subplots(1,3,figsize=(15,5))

for ion,ax in zip(ions,axes.flatten()):

    ax.step(df.index,df[ion,'full'],color='b',where='mid',label='Full')
    ax.step(df.index,df[ion,'mean'],color='r',where='mid',label='Mean')
    ax.fill_between(df.index,df[ion,'mean']+df[ion,'std'],df[ion,'mean']-df[ion,'std'],
                        color='r',alpha=0.25,step='mid')

    ax.legend(frameon=True,loc='upper right')
    ax.set_xlabel('Velocity [km/s]')
    ax.set_title(ion)

fig.tight_layout()
fig.savefig('tpcfTesting.png',dpi=300,bbox_inches='tight')

    
