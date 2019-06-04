import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math

# PSD analytical.
# Type=1 --> one-sided PSD.
# Type=2 --> two-sided PSD. 

def aLIGO_PSD(f,type):
    cutoff = -109.35 + math.log(2e10)
    logpsd=np.zeros(len(f))
    if f[0]==0:
        f[0]=f[1]
    if type == 1:
        for i in range(len(f)):
            x = f[i]/215.
            x2 = x*x
            logpsd[i] = np.log(1e-49) + np.log(x**(-4.14) -5./x2 + 111.*(1-x2+0.5*x2*x2)/(1.+0.5*x2))

            if logpsd[i]>cutoff:
                logpsd[i]=cutoff
        output=np.exp(logpsd)
    else:
        for i in range(int(len(f)/2)+1):
            x = np.abs(f[i]/215.)
            x2 = x*x
            logpsd[i] = np.log(1e-49) + np.log(x**(-4.14) -5./x2 + 111.*(1-x2+0.5*x2*x2)/(1.+0.5*x2))

            if logpsd[i]>cutoff:
                logpsd[i]=cutoff
            if i>0:
                logpsd[len(f)-i]=logpsd[i]
        output=np.exp(logpsd)/2.            # Two sided PSD
                
    return (output)


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

# Main: noise generator    
    
Fs = 20000.0
mean = 0.0
std = 1.0
n = int(10*Fs)

freq = Fs*np.fft.fftfreq(n)       # two-sided frequency vector
psd2 = aLIGO_PSD(freq,2)          # two-sided PSD
newpsd2=aLIGO_PSD_new(freq,2)     # two-sided PSD              

ff=freq[range(0,int(n/2))]        # one-sided frequency vector 
psd=aLIGO_PSD(ff,1)               # one-sided PSD              
newpsd=aLIGO_PSD_new(ff,1)        # one-sided PSD              

X = np.random.normal(mean, std, size=n)      # Gaussian white noise
XX = np.sqrt(Fs)*np.fft.fft(X)               # FFT computing and normalization
XXX = XX*np.sqrt(newpsd2)                       # Coloring
Y = np.fft.ifft(XXX)                         # FFT inverse
Y = Y.real                                   # imag part is ~ 0

YY = np.fft.fft(Y)
phase=np.angle(YY)                           # Check phase is random

# aLIGO curves
input = np.loadtxt("../PSD/aLIGODesign.txt", dtype='f', delimiter='  ')
xdata=input[:,0]
ydata=input[:,1]

# Plotting
plt.figure(1)
plt.loglog(ff,np.sqrt(psd),'r',label='Patricio ASD')
plt.loglog(ff,np.sqrt(newpsd),'b',label='aLIGO fit')
plt.loglog(xdata,ydata,'g',label='aLIGO data')
plt.xlabel('Frequency [Hz]')
plt.ylabel('ASD')
plt.grid(True)
plt.legend()

plt.figure(2)
plt.plot(freq[range(int(n/2))],phase[range(int(n/2))],'.')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase')

fig,ax=plt.subplots(2,1)
ax[0].plot(X,label='white Gaussian')
ax[1].plot(Y,label='Colored Gaussian')
ax[0].set(xlabel='Time [s]', ylabel='Strain')
ax[1].set(xlabel='Time [s]', ylabel='Strain')
ax[0].legend()
ax[1].legend()


nperseg=Fs
noverlap=nperseg/2

f,pxx=scipy.signal.welch(X, Fs, 'hanning', nperseg, noverlap)
f2,pxx2=scipy.signal.welch(Y, Fs, 'hanning', nperseg, noverlap)

fig,ax=plt.subplots(2, 1)
ax[0].plot(f,np.sqrt(pxx),label='white noise')
ax[0].set(xlabel='Frequency [Hz]', ylabel='ASD')

ax[0].grid(True)
ax[0].legend()

ax[1].loglog(f2,np.sqrt(pxx2),'k',label='Simulated noise ASD')
ax[1].loglog(ff,np.sqrt(psd),'r',label='Patricio ASD')
ax[1].loglog(ff,np.sqrt(newpsd),'b',label='aLIGO fit ASD')
ax[0].set(xlabel='Frequency [Hz]', ylabel='ASD')
ax[1].grid(True)
ax[1].legend()
plt.show()




