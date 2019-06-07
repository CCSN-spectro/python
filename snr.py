import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math
import scipy
from scipy import interpolate
from scipy.interpolate import interp1d



def aLIGO_PSD_new(f,type):
# aLIGO sensitivity curve: fit the data point from https://dcc.ligo.org/LIGO-T1800044/public
# Type=1 --> one-sided PSD.
# Type=2 --> two-sided PSD. 

    S1 = 5.0e-26
    S2 = 1.0e-40
    S3 = 1.4e-46
    S4 = 2.7e-51
    fcut = 10.
    cutoff=1e-38
    output=np.zeros(len(f))

    # to avoid issue with f=0
    if f[0]==0:
        f[0]=f[1]

    if type == 1:
        for i in range(len(f)):
            x=np.abs(f[i])
            output[i] =  S1/np.power(x, 20.) + S2/np.power(x, 4.05) + S3/np.power(x,.5) + S4*np.power (x/fcut, 2.0)
            if output[i]>cutoff:
                output[i]=cutoff
    else:
        for i in range(int(len(f)/2)+1):
            x=np.abs(f[i])
            output[i] = S1/np.power(x, 20.) + S2/np.power(x, 4.05) + S3/np.power(x,.5) + S4*np.power (x/fcut, 2.0)
            if output[i]>cutoff:
                output[i]=cutoff

            if i>0:
                output[len(f)-i]=output[i]

        output=output/2.          # Two sided PSD
                
    return output

# Main: SNR compute

#wvf = np.loadtxt("../waveforms/KURODA_SFHX_H_resampled.dat", dtype='f', delimiter=' ')
wvf = np.loadtxt("../waveforms/s20-gw_10kpc.dat", dtype='f', delimiter=' ')
#wvf = np.loadtxt("../waveforms/s20-gw_10kpc_original.dat", dtype='f', delimiter=' ')
t=wvf[:,0]
h=wvf[:,1]

dt=t[1]-t[0]
Fs=round(1./dt)
deltaf=.1
n=int(Fs/deltaf)

if len(h) < n:
    a=np.zeros(n)
    a[0:len(h)]=h
    h=a

print(dt,Fs,n,len(h))

freq=Fs*np.fft.fftfreq(n)         # two-sided frequency vector
ff=freq[range(0,int(n/2))]        # one-sided frequency vector
psd=aLIGO_PSD_new(ff,1)           # one-sided PSD

factor=[1./Fs+0.j]

hf=np.fft.fft(h)*factor           # FFT of the waveform
hfr=hf[range(int(n/2))]           # one-sided FFT of the wavefrom

integrand=np.abs(hfr*hfr.conjugate())/psd

snr = np.sqrt(4*np.trapz(integrand,ff,deltaf))
print('SNR=',snr)

snr = np.sqrt(4*integrand.sum()*deltaf)
print('SNR=',snr)




fig1 = plt.figure(1)
plt.semilogy(ff,np.sqrt(psd),'b',label='aLIGO')
plt.semilogy(ff,np.absolute(hfr),'r',label='KURODA 10kpc')
#plt.plot(ff,integrand,'g',label='integrand')
plt.xlabel('Frequency [Hz]')
plt.ylabel('ASD / h tilde ')
plt.grid
plt.show()



